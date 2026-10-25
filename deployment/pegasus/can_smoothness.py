"""
CAN 데이터 기반 실차 Smoothness 평가 스크립트.

inference.cpp --log-can 으로 기록한 CSV를 입력으로 받아
시뮬레이션의 analyze_smoothness.py와 동일한 지표를 계산합니다.

입력 CSV 형식:
  timestamp_us, speed_kmh, steering_angle_deg

자전거 모델로 유도되는 신호:
  yaw_rate  = (v_mps / WHEELBASE) * tan(steer_wheel_rad / STEER_RATIO)
  lon_acc   = d(v_mps) / dt
  lat_acc   = v_mps * yaw_rate
  yaw_acc   = d(yaw_rate) / dt

사용법:
  python can_smoothness.py can_log_20260428_103000.csv
  python can_smoothness.py can_log_20260428_103000.csv --plot
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

# ── 차량 파라미터 (inference.cpp와 동일) ─────────────────────────────────────
WHEELBASE   = 2.765   # Santafe 휠베이스 (m)
STEER_RATIO = 12.0    # 핸들/바퀴 기어비

# ── Smoothness 기준값 (Bench2Drive 동일) ─────────────────────────────────────
MAX_ABS_LAT_ACCEL = 2.93   # m/s²
MAX_LON_ACCEL     = 3.84   # m/s²
MIN_LON_ACCEL     = -8.00  # m/s²
MAX_ABS_MAG_JERK  = 8.37   # m/s³
MAX_ABS_LON_JERK  = 8.37   # m/s³
MAX_ABS_YAW_ACCEL = 1.93   # rad/s²
MAX_ABS_YAW_RATE  = 0.93   # rad/s

SEG_LEN   = 20    # 세그먼트 길이 (샘플 수), 20Hz → 1초
DT        = 0.05  # 20Hz → 0.05s


def within_bound(arr, lo, hi):
    return bool(np.all((arr >= lo) & (arr <= hi)))


def compute_smoothness_score(lon_acc, lat_acc, yaw_rate, yaw_acc, dt=DT, seg_len=SEG_LEN):
    """
    전체 구간에 대해 세그먼트 단위 smoothness 계산 후 평균.
    통과 비율 (0~1, 높을수록 부드러움).
    """
    n = len(lon_acc)
    ws = min(7, seg_len)
    kw  = dict(polyorder=2, window_length=ws, axis=-1)
    kw2 = dict(polyorder=2, window_length=ws, deriv=1, delta=dt, axis=-1)

    seg_scores = []
    fail_counts = {k: 0 for k in ['lon_acc','lat_acc','mag_jerk','lon_jerk','yaw_acc','yaw_rate']}
    total_segs  = 0

    for i in range(n // seg_len):
        sl = slice(i * seg_len, (i + 1) * seg_len)
        if sl.stop > n:
            continue

        la  = savgol_filter(lon_acc[sl],  **kw)
        lta = savgol_filter(lat_acc[sl],  **kw)
        ya  = savgol_filter(yaw_acc[sl],  **kw)
        yr  = savgol_filter(yaw_rate[sl], **kw)
        mag = np.hypot(la, lta)

        checks = {
            'lon_acc':  within_bound(la,  MIN_LON_ACCEL,     MAX_LON_ACCEL),
            'lat_acc':  within_bound(lta, -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
            'mag_jerk': within_bound(savgol_filter(mag, **kw2), -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
            'lon_jerk': within_bound(savgol_filter(la,  **kw2), -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
            'yaw_acc':  within_bound(ya,  -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
            'yaw_rate': within_bound(yr,  -MAX_ABS_YAW_RATE,  MAX_ABS_YAW_RATE),
        }
        seg_ok = sum(checks.values()) / len(checks)
        seg_scores.append(seg_ok)
        total_segs += 1
        for k, ok in checks.items():
            if not ok:
                fail_counts[k] += 1

    smoothness = float(np.mean(seg_scores)) if seg_scores else 0.0
    fail_rates = {k: fail_counts[k] / total_segs * 100 if total_segs else 0.0
                  for k in fail_counts}
    return smoothness, fail_rates, total_segs


def load_and_derive(csv_path: str):
    """CSV 로드 후 자전거 모델로 신호 유도."""
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ts_us   = data[:, 0]
    spd_kmh = data[:, 1]
    steer_deg = data[:, 2]

    # 실제 dt (타임스탬프 기반)
    dt_arr = np.diff(ts_us) * 1e-6  # us → s
    dt_arr = np.clip(dt_arr, 0.001, 0.5)  # 이상값 클램핑
    dt_med = float(np.median(dt_arr))
    print(f"  샘플 수: {len(spd_kmh)}  중앙 dt: {dt_med*1000:.1f}ms ({1/dt_med:.1f}Hz)")

    v_mps      = spd_kmh / 3.6
    steer_rad  = np.radians(steer_deg / STEER_RATIO)  # 바퀴 조향각

    # 자전거 모델: yaw_rate
    yaw_rate = (v_mps / WHEELBASE) * np.tan(steer_rad)

    # 종방향 가속도: d(v)/dt (중앙차분)
    lon_acc = np.gradient(v_mps, dt_med)

    # 횡방향 가속도: v * yaw_rate
    lat_acc = v_mps * yaw_rate

    # 요가속도: d(yaw_rate)/dt
    yaw_acc = np.gradient(yaw_rate, dt_med)

    return lon_acc, lat_acc, yaw_rate, yaw_acc, dt_med, len(v_mps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="can_log_YYYYMMDD_HHMMSS.csv 경로")
    parser.add_argument("--plot", action="store_true", help="신호 그래프 출력")
    parser.add_argument("--seg-len", type=int, default=SEG_LEN,
                        help=f"세그먼트 길이 샘플 수 (기본: {SEG_LEN} = 1초@20Hz)")
    args = parser.parse_args()

    seg_len = args.seg_len

    print(f"\n[입력] {args.csv}")
    lon_acc, lat_acc, yaw_rate, yaw_acc, dt, n_samples = load_and_derive(args.csv)

    smoothness, fail_rates, n_segs = compute_smoothness_score(lon_acc, lat_acc, yaw_rate, yaw_acc, dt, seg_len)

    print(f"\n{'='*55}")
    print(f"  Smoothness Score : {smoothness:.4f}  ({n_segs} 세그먼트)")
    print(f"{'='*55}")
    print(f"  {'Metric':<14} {'Fail Rate':>10}  {'기준'}")
    print(f"  {'-'*50}")
    bounds = {
        'lon_acc':  f"[{MIN_LON_ACCEL:.1f}, {MAX_LON_ACCEL:.1f}] m/s²",
        'lat_acc':  f"[±{MAX_ABS_LAT_ACCEL:.2f}] m/s²",
        'mag_jerk': f"[±{MAX_ABS_MAG_JERK:.2f}] m/s³",
        'lon_jerk': f"[±{MAX_ABS_LON_JERK:.2f}] m/s³",
        'yaw_acc':  f"[±{MAX_ABS_YAW_ACCEL:.2f}] rad/s²",
        'yaw_rate': f"[±{MAX_ABS_YAW_RATE:.2f}] rad/s",
    }
    for k, fr in fail_rates.items():
        print(f"  {k:<14} {fr:>8.1f}%   {bounds[k]}")
    print(f"{'='*55}\n")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            t = np.arange(n_samples) * dt

            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            axes[0].plot(t, lon_acc, label='lon_acc (m/s²)')
            axes[0].axhline(MAX_LON_ACCEL,  color='r', ls='--', lw=0.8)
            axes[0].axhline(MIN_LON_ACCEL,  color='r', ls='--', lw=0.8)
            axes[0].set_ylabel('lon_acc'); axes[0].legend(); axes[0].grid(True)

            axes[1].plot(t, lat_acc, label='lat_acc (m/s²)', color='orange')
            axes[1].axhline( MAX_ABS_LAT_ACCEL, color='r', ls='--', lw=0.8)
            axes[1].axhline(-MAX_ABS_LAT_ACCEL, color='r', ls='--', lw=0.8)
            axes[1].set_ylabel('lat_acc'); axes[1].legend(); axes[1].grid(True)

            axes[2].plot(t, yaw_rate, label='yaw_rate (rad/s)', color='green')
            axes[2].axhline( MAX_ABS_YAW_RATE, color='r', ls='--', lw=0.8)
            axes[2].axhline(-MAX_ABS_YAW_RATE, color='r', ls='--', lw=0.8)
            axes[2].set_ylabel('yaw_rate'); axes[2].legend(); axes[2].grid(True)

            axes[3].plot(t, yaw_acc, label='yaw_acc (rad/s²)', color='purple')
            axes[3].axhline( MAX_ABS_YAW_ACCEL, color='r', ls='--', lw=0.8)
            axes[3].axhline(-MAX_ABS_YAW_ACCEL, color='r', ls='--', lw=0.8)
            axes[3].set_ylabel('yaw_acc'); axes[3].set_xlabel('time (s)')
            axes[3].legend(); axes[3].grid(True)

            fig.suptitle(f"Smoothness: {smoothness:.4f}  |  {Path(args.csv).name}")
            plt.tight_layout()
            out_png = Path(args.csv).with_suffix('.png')
            plt.savefig(out_png, dpi=120)
            print(f"[그래프] {out_png}")
            plt.show()
        except ImportError:
            print("[경고] matplotlib 없음, --plot 스킵")


if __name__ == "__main__":
    main()
