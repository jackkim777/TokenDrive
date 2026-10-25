"""
실차 자율주행 평가 스크립트 — MTBI / MDBI / Task SR

inference_v3 --log-can --rtk-port /dev/ttyUSB0 로 생성된 CSV를 분석합니다.

CSV 컬럼:
  timestamp_us, speed_kmh, steering_angle_deg,
  override, hlc, rtk_lat, rtk_lon, rtk_valid

지표:
  MTBI  : Mean Time Between Interventions (초)
  MDBI  : Mean Distance Between Interventions (m, RTK 기반)
  SR    : Task Success Rate (%) — HLC 기반 태스크별

개입(Intervention) 유형:
  MANUAL_TAKEOVER  : override 1→0 (인간이 수동으로 전환)
  TASK_ABORT       : 태스크 수행 중 개입 발생

태스크 정의:
  HLC 1  : 교차로 좌회전
  HLC 2  : 교차로 우회전
  HLC 3  : 교차로 직진
  HLC 5  : 좌측 차선변경
  HLC 6  : 우측 차선변경
  HLC 4  : lane_follow 중 완만한 커브 (곡률 임계값으로 자동 감지)

태스크 성공 기준:
  - 태스크 HLC 시작 → FOLLOW_LANE 복귀 사이에 개입 없음
  - lane_follow 커브: 커브 구간 전체에 개입 없음

사용법:
  python eval_metrics.py can_log_20260428_103000.csv
  python eval_metrics.py can_log_20260428_103000.csv --plot
  python eval_metrics.py can_log_20260428_103000.csv --min-task-dist 10
"""

import argparse
import math
import numpy as np
from pathlib import Path


# ── HLC 정의 ─────────────────────────────────────────────────────────────────
HLC_LEFT         = 1
HLC_RIGHT        = 2
HLC_STRAIGHT     = 3
HLC_FOLLOW_LANE  = 4
HLC_CHANGE_LEFT  = 5
HLC_CHANGE_RIGHT = 6

HLC_NAME = {
    HLC_LEFT:        "intersection_left",
    HLC_RIGHT:       "intersection_right",
    HLC_STRAIGHT:    "intersection_straight",
    HLC_FOLLOW_LANE: "lane_follow",
    HLC_CHANGE_LEFT: "lane_change_left",
    HLC_CHANGE_RIGHT:"lane_change_right",
}

TASK_HLCS = [HLC_LEFT, HLC_RIGHT, HLC_STRAIGHT, HLC_CHANGE_LEFT, HLC_CHANGE_RIGHT]

# lane_follow 커브 감지 임계값 (rad/s)
CURVE_YAW_RATE_THRESH = 0.08   # 이 이상이면 커브로 판단
CURVE_MIN_DURATION_S  = 2.0    # 최소 커브 지속 시간 (초)

# 태스크 최소 거리 (m) — 이보다 짧은 태스크는 무시
MIN_TASK_DIST_M = 5.0

WHEELBASE   = 2.765
STEER_RATIO = 12.0


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """두 WGS84 좌표 사이의 거리 (m)."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cum_distance(lats, lons, rtk_valid):
    """누적 이동 거리 배열 (m). rtk_valid==0인 구간은 직전 값 유지."""
    dist = np.zeros(len(lats))
    for i in range(1, len(lats)):
        if rtk_valid[i] > 0 and rtk_valid[i-1] > 0:
            dist[i] = dist[i-1] + haversine(lats[i-1], lons[i-1], lats[i], lons[i])
        else:
            dist[i] = dist[i-1]
    return dist


def yaw_rate_from_can(speed_kmh, steer_deg):
    """자전거 모델로 yaw_rate 추정 (rad/s)."""
    v = speed_kmh / 3.6
    wheel_rad = math.radians(steer_deg / STEER_RATIO)
    return (v / WHEELBASE) * math.tan(wheel_rad)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_csv(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, filling_values=0.0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    cols = ['timestamp_us','speed_kmh','steering_angle_deg',
            'override','hlc','rtk_lat','rtk_lon','rtk_valid']
    d = {c: data[:, i] for i, c in enumerate(cols)}
    d['timestamp_s'] = (d['timestamp_us'] - d['timestamp_us'][0]) * 1e-6
    d['hlc'] = d['hlc'].astype(int)
    d['override'] = d['override'].astype(int)
    d['rtk_valid'] = d['rtk_valid'].astype(int)
    d['yaw_rate'] = np.array([
        yaw_rate_from_can(s, st)
        for s, st in zip(d['speed_kmh'], d['steering_angle_deg'])
    ])
    d['cum_dist'] = cum_distance(d['rtk_lat'], d['rtk_lon'], d['rtk_valid'])
    return d, len(data)


# ── 개입 감지 ─────────────────────────────────────────────────────────────────

def detect_interventions(d):
    """
    override 1→0 전환을 개입으로 감지.
    Returns: list of dict with keys: idx, timestamp_s, cum_dist_m, type, hlc_at_event
    """
    interventions = []
    ovr = d['override']
    for i in range(1, len(ovr)):
        if ovr[i-1] == 1 and ovr[i] == 0:
            interventions.append({
                'idx':         i,
                'timestamp_s': d['timestamp_s'][i],
                'cum_dist_m':  d['cum_dist'][i],
                'type':        'MANUAL_TAKEOVER',
                'hlc':         d['hlc'][i],
                'hlc_name':    HLC_NAME.get(d['hlc'][i], f"hlc_{d['hlc'][i]}"),
                'speed_kmh':   d['speed_kmh'][i],
            })
    return interventions


# ── MTBI / MDBI 계산 ─────────────────────────────────────────────────────────

def compute_mtbi_mdbi(d, interventions):
    """
    자율주행 구간(override=1)을 추출하고
    각 자율 구간의 지속 시간 / 이동 거리를 기반으로 MTBI / MDBI 계산.
    """
    ovr = d['override']
    ts  = d['timestamp_s']
    cd  = d['cum_dist']

    # 자율 구간 [start_idx, end_idx] 추출
    auto_segments = []
    in_auto = False
    seg_start = 0
    for i in range(len(ovr)):
        if ovr[i] == 1 and not in_auto:
            in_auto   = True
            seg_start = i
        elif ovr[i] == 0 and in_auto:
            in_auto = False
            auto_segments.append((seg_start, i - 1))
    if in_auto:
        auto_segments.append((seg_start, len(ovr) - 1))

    durations = [ts[e] - ts[s]  for s, e in auto_segments]
    distances = [cd[e] - cd[s]  for s, e in auto_segments]

    total_auto_time  = sum(durations)
    total_auto_dist  = sum(distances)
    n_interventions  = len(interventions)

    mtbi = total_auto_time  / n_interventions if n_interventions > 0 else float('inf')
    mdbi = total_auto_dist  / n_interventions if n_interventions > 0 else float('inf')

    return {
        'mtbi_s':          mtbi,
        'mdbi_m':          mdbi,
        'n_interventions': n_interventions,
        'total_auto_time_s':  total_auto_time,
        'total_auto_dist_m':  total_auto_dist,
        'n_auto_segments':    len(auto_segments),
    }


# ── Task SR 계산 ─────────────────────────────────────────────────────────────

def detect_tasks(d, min_dist_m=MIN_TASK_DIST_M):
    """
    HLC 기반 태스크 구간 + lane_follow 커브 구간 탐지.
    Returns: list of task dicts
    """
    tasks = []
    hlc   = d['hlc']
    ovr   = d['override']
    ts    = d['timestamp_s']
    cd    = d['cum_dist']
    yr    = d['yaw_rate']
    n     = len(hlc)

    # ── 1. HLC 태스크 (non-follow_lane) ───────────────────────────────────
    i = 0
    while i < n:
        if ovr[i] == 1 and hlc[i] in TASK_HLCS:
            task_hlc = hlc[i]
            start_i  = i
            # 태스크 끝: HLC가 FOLLOW_LANE으로 돌아오거나 override 꺼지거나 끝
            while i < n and hlc[i] == task_hlc and ovr[i] == 1:
                i += 1
            end_i = i - 1

            dist = cd[end_i] - cd[start_i]
            if dist < min_dist_m:
                continue  # 너무 짧은 구간 무시

            # 개입 여부: 태스크 구간 내 override 0이 있으면 실패
            intervened = bool(np.any(ovr[start_i:end_i+1] == 0))
            # 태스크 종료 방식 확인
            if end_i + 1 < n:
                completed = (hlc[end_i+1] == HLC_FOLLOW_LANE)
            else:
                completed = False

            tasks.append({
                'type':        HLC_NAME[task_hlc],
                'hlc':         task_hlc,
                'start_s':     ts[start_i],
                'end_s':       ts[end_i],
                'duration_s':  ts[end_i] - ts[start_i],
                'dist_m':      dist,
                'intervened':  intervened,
                'completed':   completed and not intervened,
                'start_dist_m': cd[start_i],
            })
        else:
            i += 1

    # ── 2. lane_follow 중 완만한 커브 감지 ────────────────────────────────
    in_curve  = False
    curve_start = 0
    for i in range(n):
        is_curve = (abs(yr[i]) >= CURVE_YAW_RATE_THRESH and
                    hlc[i] == HLC_FOLLOW_LANE and ovr[i] == 1)
        if is_curve and not in_curve:
            in_curve    = True
            curve_start = i
        elif not is_curve and in_curve:
            in_curve = False
            duration = ts[i-1] - ts[curve_start]
            dist     = cd[i-1] - cd[curve_start]
            if duration >= CURVE_MIN_DURATION_S and dist >= min_dist_m:
                intervened = bool(np.any(ovr[curve_start:i] == 0))
                tasks.append({
                    'type':        'lane_follow_curve',
                    'hlc':         HLC_FOLLOW_LANE,
                    'start_s':     ts[curve_start],
                    'end_s':       ts[i-1],
                    'duration_s':  duration,
                    'dist_m':      dist,
                    'intervened':  intervened,
                    'completed':   not intervened,
                    'start_dist_m': cd[curve_start],
                })

    tasks.sort(key=lambda x: x['start_s'])
    return tasks


def compute_sr(tasks):
    """태스크 유형별 Success Rate."""
    from collections import defaultdict
    counts  = defaultdict(int)
    success = defaultdict(int)
    for t in tasks:
        counts[t['type']] += 1
        if t['completed']:
            success[t['type']] += 1
    sr = {}
    for k in counts:
        sr[k] = success[k] / counts[k] * 100 if counts[k] > 0 else 0.0
    return sr, counts, success


# ── 출력 ─────────────────────────────────────────────────────────────────────

def print_report(mtbi_mdbi, sr, counts, success, interventions, tasks, n_samples, duration_s):
    W = 60
    print("\n" + "=" * W)
    print("  실차 자율주행 평가 결과")
    print("=" * W)
    print(f"  총 샘플 수  : {n_samples}")
    print(f"  총 시간     : {duration_s:.1f} s  ({duration_s/60:.1f} min)")
    print(f"  자율 시간   : {mtbi_mdbi['total_auto_time_s']:.1f} s")
    print(f"  자율 거리   : {mtbi_mdbi['total_auto_dist_m']:.1f} m")
    print(f"  자율 구간 수: {mtbi_mdbi['n_auto_segments']}")
    print()

    print(f"  ── MTBI / MDBI {'─'*35}")
    n = mtbi_mdbi['n_interventions']
    if n == 0:
        print(f"  개입 없음 ✓  (전체 구간 개입 0회)")
    else:
        mtbi = mtbi_mdbi['mtbi_s']
        mdbi = mtbi_mdbi['mdbi_m']
        print(f"  개입 횟수  : {n} 회")
        print(f"  MTBI       : {mtbi:.1f} s  ({mtbi/60:.2f} min)")
        print(f"  MDBI       : {mdbi:.1f} m  ({mdbi/1000:.3f} km)")

    print()
    print(f"  ── 개입 상세 {'─'*38}")
    if not interventions:
        print("  없음")
    for iv in interventions:
        print(f"  [{iv['timestamp_s']:7.1f}s | {iv['cum_dist_m']:6.0f}m]  "
              f"{iv['type']:<20}  HLC:{iv['hlc_name']}  "
              f"속도:{iv['speed_kmh']:.1f}km/h")

    print()
    print(f"  ── Task Success Rate {'─'*31}")
    task_order = [
        'lane_change_left', 'lane_change_right',
        'intersection_straight', 'intersection_left', 'intersection_right',
        'lane_follow_curve',
    ]
    all_types = task_order + [k for k in counts if k not in task_order]
    for t in all_types:
        if t not in counts:
            continue
        n_t = counts[t]
        s_t = success[t]
        rate = sr[t]
        bar  = '█' * int(rate / 5) + '░' * (20 - int(rate / 5))
        print(f"  {t:<24} {bar}  {s_t:2d}/{n_t:2d} ({rate:5.1f}%)")

    print()
    print(f"  ── 태스크 상세 {'─'*36}")
    print(f"  {'#':<3} {'type':<24} {'dur':>5}s {'dist':>6}m  결과")
    print(f"  {'-'*58}")
    for i, t in enumerate(tasks):
        result = "✓ 성공" if t['completed'] else "✗ 개입"
        print(f"  {i+1:<3} {t['type']:<24} {t['duration_s']:5.1f}  "
              f"{t['dist_m']:6.0f}m  {result}")

    print("=" * W + "\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv",             help="can_log_*.csv 경로")
    parser.add_argument("--plot",          action="store_true")
    parser.add_argument("--min-task-dist", type=float, default=MIN_TASK_DIST_M,
                        help=f"태스크 최소 거리 m (기본: {MIN_TASK_DIST_M})")
    args = parser.parse_args()

    print(f"\n[입력] {args.csv}")
    d, n_samples = load_csv(args.csv)
    duration_s   = d['timestamp_s'][-1] - d['timestamp_s'][0]

    interventions = detect_interventions(d)
    mtbi_mdbi     = compute_mtbi_mdbi(d, interventions)
    tasks         = detect_tasks(d, min_dist_m=args.min_task_dist)
    sr, counts, success = compute_sr(tasks)

    print_report(mtbi_mdbi, sr, counts, success, interventions, tasks, n_samples, duration_s)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            ts = d['timestamp_s']

            # override + HLC
            ax = axes[0]
            ax.fill_between(ts, d['override'], alpha=0.3, color='green', label='override(auto)')
            ax2 = ax.twinx()
            ax2.plot(ts, d['hlc'], color='navy', lw=0.8, label='HLC')
            ax2.set_yticks(list(HLC_NAME.keys()))
            ax2.set_yticklabels([HLC_NAME[k][:12] for k in HLC_NAME], fontsize=7)
            ax.set_ylabel('Override'); ax.legend(loc='upper left')
            ax.set_title('Override & HLC'); ax.grid(True, alpha=0.3)

            # 개입 마킹
            for iv in interventions:
                ax.axvline(iv['timestamp_s'], color='red', lw=1.5, alpha=0.8)

            # 속도
            ax = axes[1]
            ax.plot(ts, d['speed_kmh'], color='steelblue', lw=0.8, label='speed (km/h)')
            ax.axhline(15, color='red', ls='--', lw=0.8, label='15km/h limit')
            for iv in interventions:
                ax.axvline(iv['timestamp_s'], color='red', lw=1.5, alpha=0.8)
            # 태스크 구간 색칠
            colors = {'lane_change_left':'#FFD700','lane_change_right':'#FFA500',
                      'intersection_left':'#90EE90','intersection_right':'#32CD32',
                      'intersection_straight':'#00CED1','lane_follow_curve':'#DDA0DD'}
            for t in tasks:
                c = colors.get(t['type'], '#CCCCCC')
                ax.axvspan(t['start_s'], t['end_s'], alpha=0.25, color=c)
            ax.set_ylabel('Speed (km/h)'); ax.legend(); ax.grid(True, alpha=0.3)

            # RTK 궤적
            ax = axes[2]
            valid = d['rtk_valid'] > 0
            if np.any(valid):
                sc = ax.scatter(d['rtk_lon'][valid], d['rtk_lat'][valid],
                                c=ts[valid], cmap='viridis', s=2)
                plt.colorbar(sc, ax=ax, label='time (s)')
                for iv in interventions:
                    idx = iv['idx']
                    if d['rtk_valid'][idx] > 0:
                        ax.plot(d['rtk_lon'][idx], d['rtk_lat'][idx],
                                'rv', ms=8, zorder=5)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.set_title('RTK Trajectory (▼=개입 위치)'); ax.grid(True, alpha=0.3)

            # SR 막대
            fig2, ax3 = plt.subplots(figsize=(10, 4))
            labels = list(sr.keys())
            values = [sr[k] for k in labels]
            bars   = ax3.bar(labels, values, color=['#4CAF50' if v >= 80 else '#FF7043' for v in values])
            ax3.set_ylim(0, 110)
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Task Success Rate')
            ax3.axhline(100, color='gray', ls='--', lw=0.8)
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, val + 1,
                         f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
            plt.xticks(rotation=20, ha='right')
            plt.tight_layout()

            out1 = Path(args.csv).with_suffix('') + '_eval.png'
            out2 = Path(args.csv).with_suffix('') + '_sr.png'
            fig.savefig(str(out1), dpi=120, bbox_inches='tight')
            fig2.savefig(str(out2), dpi=120, bbox_inches='tight')
            print(f"[그래프] {out1}")
            print(f"[그래프] {out2}")
            plt.show()
        except ImportError:
            print("[경고] matplotlib 없음, --plot 스킵")


if __name__ == "__main__":
    main()
