/**
 * TokenDrive Inference + GMSL Camera + GUI Visualization (C++)
 *
 * ONNX Runtime + DriveWorks GMSL 카메라 + GL 시각화
 * Pegasus (aarch64 / CUDA 10.2 / EGL)에서 실행.
 *
 * 빌드:
 *   mkdir build && cd build
 *   cmake .. -DORT_INCLUDE_DIR=/path/to/ort/include -DORT_LIBRARY=/path/to/libonnxruntime.so
 *   make -j4
 *
 * 실행:
 *   ./inference_v3 --model tokendrive_fp32_wp64_op15_v3_token_on.onnx
 */

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <fstream>
#include <sys/stat.h>

// SocketCAN
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "preprocess_cuda.cuh"
#endif

// DriveWorks core
#include <dw/core/Context.h>
#include <dw/core/EGL.h>
#include <dw/core/VersionCurrent.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>

// DriveWorks visualization
#include <dwvisualization/core/Visualization.h>
#include <dwvisualization/core/Renderer.h>
#include <dwvisualization/interop/ImageStreamer.h>
#include <dwvisualization/gl/GL.h>

// GLFW
#define GLFW_INCLUDE_ES3
#define GLFW_EXPOSE_NATIVE_EGL
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <EGL/eglext.h>

// ── 상수 ───────────────────────────────────────────────────────────────────────

static const int   INPUT_H       = 448;
static const int   INPUT_W       = 448;
static const int   NUM_SPEED_WPS = 10;
static const int   NUM_ROUTE_WPS = 64;
static const int   WIN_W         = 1280;
static const int   WIN_H         = 800;

static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

static const int CAMERA_VIEW_BASELINE = 0;
static const int CAMERA_VIEW_FTHETA_RECTILINEAR = 1;

static const float BONNET_CX = 964.875977f;
static const float BONNET_CY = 645.572449f;
static const float BONNET_WIDTH = 1920.0f;
static const float BONNET_HEIGHT = 1208.0f;
static const float BONNET_CAM_X = 1.77f;  // rig FLU, meters
static const float BONNET_CAM_Y = 0.0f;
static const float BONNET_CAM_Z = 1.47f;
static const float SIM_HFOV_DEG = 110.0f;
static const float SIM_VFOV_DEG = 71.0753555839f;  // from 1024x512, HFOV=110

enum HLC : int64_t {
    HLC_LEFT = 1, HLC_RIGHT = 2, HLC_STRAIGHT = 3,
    HLC_FOLLOW_LANE = 4, HLC_CHANGE_LEFT = 5, HLC_CHANGE_RIGHT = 6,
};

static volatile bool g_running = true;
static std::atomic<bool> g_log_running{false};
static void signal_handler(int) { g_running = false; g_log_running.store(false); }

static const char* SAVE_ROOT_DIR =
    "./save_frames";

static bool ensure_directory(const std::string& path)
{
    struct stat st {};
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode))
            return true;
        std::cerr << "[save] path exists but is not a directory: " << path << "\n";
        return false;
    }

    if (mkdir(path.c_str(), 0755) == 0)
        return true;

    if (errno == EEXIST && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        return true;

    std::cerr << "[save] failed to create directory: " << path
              << " (" << std::strerror(errno) << ")\n";
    return false;
}

static bool verify_directory_writable(const std::string& path)
{
    const std::string probe = path + "/.write_test";
    std::ofstream ofs(probe, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "[save] directory is not writable: " << path << "\n";
        return false;
    }
    ofs << "ok\n";
    ofs.close();
    std::remove(probe.c_str());
    return true;
}

// 키보드로 HLC 변경 (글로벌)
static int64_t g_hlc = HLC_FOLLOW_LANE;

// CAN 차량 속도 (km/h), 조향각 (deg) — CAN 스레드에서 갱신
static std::atomic<float> g_vehicle_speed_kmh{0.0f};
static std::atomic<float> g_steering_angle_deg{0.0f};
static std::atomic<int>   g_switch_state{0};       // 0x51 byte7: bit1(2)=Auto, bit2(4)=APM, bit3(8)=ASM, bit5(32)=AGM
static std::atomic<int>   g_override_feedback{0};  // 0x52 byte0: Override_feedback

// ── Token 레이블 ──────────────────────────────────────────────────────────────
static const char* LAT_ACTION_LABELS[]    = {"other", "go_around", "overtake", "give_way"};
static const char* LON_ACTION_LABELS[]    = {"other", "remain_stop", "stop_now", "slow_down",
                                             "maintain", "maint_red", "increase", "wait_gap"};
static const char* CTX_SPEED_KIND_LABELS[] = {"none", "dynamic", "static", "policy"};
static const char* CTX_SPEED_SUB_LABELS[]  = {"NA", "vehicle", "pedestrian", "stop_sign",
                                              "traffic_light", "construction"};
static const char* CTX_ROUTE_KIND_LABELS[] = {"none", "static", "policy"};
static const char* CTX_ROUTE_SUB_LABELS[]  = {"NA", "construction", "lane_change",
                                              "turn", "route_adj"};

static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

static std::vector<float> softmax(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    float mx = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) { out[i] = std::exp(v[i] - mx); sum += out[i]; }
    for (auto& x : out) x /= sum;
    return out;
}

// ── Waypoint (forward declaration) ────────────────────────────────────────────
struct Waypoint { float x, y; };

// ── 제어 모드 ────────────────────────────────────────────────────────────────
enum CtrlMode { MODE_NONE = 0, MODE_PP_STEER, MODE_PP_FULL, MODE_ST_STEER, MODE_ST_FULL };
static CtrlMode g_ctrl_mode = MODE_NONE;
static std::atomic<bool>  g_override{false};          // O키 토글: true=자율, false=수동

// 제어 출력값 (메인 루프에서 갱신, CAN 송신 스레드에서 읽기)
static std::atomic<float> g_cmd_steer_deg{0.0f};      // 조향 명령 (deg)
static std::atomic<int>   g_cmd_accel{650};            // 엑셀 명령 [650~3400]
static std::atomic<int>   g_cmd_brake{0};              // 브레이크 명령 [0~17000]
static std::atomic<int>   g_cmd_gear{5};               // 기어 (5=D)
static std::atomic<int>   g_cmd_angular_speed{50};      // 조향 속도 명령 [0~255]

// ── 차량 & 튜닝 파라미터 (키보드로 실시간 조절 가능) ─────────────────────────
static const float WHEELBASE   = 2.765f;   // Santafe 휠베이스 (m)
static const float STEER_RATIO = 12.0f;    // 핸들/바퀴 기어비

// Speed safety limits for full control modes.
static constexpr float SPEED_CAP_KMH = 15.0f;
static constexpr float SAFETY_BRAKE_ON_KMH = 15.0f;
static constexpr float SAFETY_BRAKE_OFF_KMH = 13.0f;
static constexpr int SAFETY_BRAKE_CMD = 7000;

// Pure Pursuit
static std::atomic<float> g_pp_ld_min{7.0f};     // look-ahead 최소 (m)
static std::atomic<float> g_pp_ld_max{15.0f};    // look-ahead 최대 (m)
static std::atomic<float> g_pp_ld_gain{2.5f};    // speed * gain = ld

// Stanley
static std::atomic<float> g_st_k_gain{2.5f};     // cross-track gain
static std::atomic<float> g_st_k_soft{0.5f};     // softening constant

// 튜닝 모드: T키로 대상 전환, UP/DOWN으로 값 조절
enum TuneTarget { TUNE_PP_LD_MIN, TUNE_PP_LD_MAX, TUNE_PP_LD_GAIN,
                  TUNE_ST_K_GAIN, TUNE_ST_K_SOFT, TUNE_STEER_SPEED, TUNE_COUNT };
static TuneTarget g_tune_target = TUNE_PP_LD_MIN;

static const char* tune_name(TuneTarget t) {
    switch (t) {
        case TUNE_PP_LD_MIN:   return "PP ld_min";
        case TUNE_PP_LD_MAX:   return "PP ld_max";
        case TUNE_PP_LD_GAIN:  return "PP ld_gain";
        case TUNE_ST_K_GAIN:   return "ST k_gain";
        case TUNE_ST_K_SOFT:   return "ST k_soft";
        case TUNE_STEER_SPEED: return "Steer speed";
        default: return "?";
    }
}

static float tune_get(TuneTarget t) {
    switch (t) {
        case TUNE_PP_LD_MIN:   return g_pp_ld_min.load();
        case TUNE_PP_LD_MAX:   return g_pp_ld_max.load();
        case TUNE_PP_LD_GAIN:  return g_pp_ld_gain.load();
        case TUNE_ST_K_GAIN:   return g_st_k_gain.load();
        case TUNE_ST_K_SOFT:   return g_st_k_soft.load();
        case TUNE_STEER_SPEED: return static_cast<float>(g_cmd_angular_speed.load());
        default: return 0;
    }
}

static void tune_adjust(TuneTarget t, float delta) {
    switch (t) {
        case TUNE_PP_LD_MIN:   g_pp_ld_min.store(std::max(1.0f, g_pp_ld_min.load() + delta)); break;
        case TUNE_PP_LD_MAX:   g_pp_ld_max.store(std::max(2.0f, g_pp_ld_max.load() + delta)); break;
        case TUNE_PP_LD_GAIN:  g_pp_ld_gain.store(std::max(0.1f, g_pp_ld_gain.load() + delta)); break;
        case TUNE_ST_K_GAIN:   g_st_k_gain.store(std::max(0.1f, g_st_k_gain.load() + delta)); break;
        case TUNE_ST_K_SOFT:   g_st_k_soft.store(std::max(0.1f, g_st_k_soft.load() + delta)); break;
        case TUNE_STEER_SPEED: {
            int step = (delta >= 0.0f) ? 5 : -5;
            int next = std::max(0, std::min(255, g_cmd_angular_speed.load() + step));
            g_cmd_angular_speed.store(next);
            break;
        }
        default: break;
    }
}

// ── RTK GPS (NovAtel OEM7 via TCP ICOM2) ─────────────────────────────────────

static std::atomic<double> g_rtk_lat{0.0};
static std::atomic<double> g_rtk_lon{0.0};
static std::atomic<int>    g_rtk_quality{0};  // 0=no fix, 1=GPS, 2=DGPS, 4=RTK int, 5=RTK float

static double nmea_to_deg(const std::string& val, const std::string& hemi) {
    if (val.empty()) return 0.0;
    double raw = std::stod(val);
    int deg = static_cast<int>(raw / 100);
    double min = raw - deg * 100.0;
    double dd = deg + min / 60.0;
    if (hemi == "S" || hemi == "W") dd = -dd;
    return dd;
}

static void parse_gpgga(const std::string& line) {
    std::vector<std::string> f;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) f.push_back(tok);
    if (f.size() < 7) return;
    try {
        int quality = std::stoi(f[6]);
        double lat = nmea_to_deg(f[2], f[3]);
        double lon = nmea_to_deg(f[4], f[5]);
        g_rtk_lat.store(lat);
        g_rtk_lon.store(lon);
        g_rtk_quality.store(quality);
    } catch (...) {}
}

static void rtk_tcp_thread(const std::string& ip, int port)
{
    while (g_running) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "[RTK] 소켓 생성 실패\n";
            sleep(3);
            continue;
        }
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

        if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[RTK] 연결 실패: " << ip << ":" << port << "\n";
            close(sock);
            sleep(3);
            continue;
        }

        std::string cmd = "LOG GPGGA ONTIME 0.1\r\n";
        write(sock, cmd.c_str(), cmd.size());
        std::cout << "[RTK] " << ip << ":" << port << " 연결, GPGGA 수신 시작\n";

        char buf[1024];
        std::string accum;
        while (g_running) {
            int n = recv(sock, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;
            buf[n] = '\0';
            accum += buf;
            size_t pos;
            while ((pos = accum.find('\n')) != std::string::npos) {
                std::string sentence = accum.substr(0, pos);
                accum.erase(0, pos + 1);
                if (sentence.find("$GPGGA") != std::string::npos ||
                    sentence.find("$GNGGA") != std::string::npos)
                    parse_gpgga(sentence);
            }
        }
        close(sock);
        if (g_running)
            std::cerr << "[RTK] 연결 끊김, 재연결 시도...\n";
    }
    std::cout << "[RTK] 수신 종료\n";
}

// ── CAN 로거 스레드 (20Hz CSV) ───────────────────────────────────────────────
static void can_logger_thread(const std::string& log_path)
{
    std::ofstream ofs(log_path);
    if (!ofs.is_open()) {
        std::cerr << "[LOG] 파일 열기 실패: " << log_path << "\n";
        return;
    }
    ofs << "timestamp_us,speed_kmh,steering_angle_deg,"
           "override,hlc,rtk_lat,rtk_lon,rtk_quality,"
           "switch_state,override_feedback\n";
    std::cout << "[LOG] 기록 시작: " << log_path << "\n";

    using Clock = std::chrono::high_resolution_clock;
    while (g_log_running.load(std::memory_order_relaxed)) {
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now().time_since_epoch()).count();
        ofs << now_us                                                    << ","
            << g_vehicle_speed_kmh.load(std::memory_order_relaxed)      << ","
            << g_steering_angle_deg.load(std::memory_order_relaxed)     << ","
            << (g_override.load(std::memory_order_relaxed) ? 1 : 0)     << ","
            << static_cast<int>(g_hlc)                                  << ","
            << std::fixed << std::setprecision(8)
            << g_rtk_lat.load()                                         << ","
            << g_rtk_lon.load()                                         << ","
            << g_rtk_quality.load()                                     << ","
            << g_switch_state.load(std::memory_order_relaxed)              << ","
            << g_override_feedback.load(std::memory_order_relaxed)         << "\n";
        usleep(50000);  // 20Hz
    }
    ofs.flush();
    std::cout << "[LOG] 기록 종료: " << log_path << "\n";
}

// ── CAN 읽기 스레드 ───────────────────────────────────────────────────────────
// Santafe DBC:
//   0x52 (82) Vehicle_Info_2: Vehicle_Speed  @ byte1-2, 16bit LE unsigned, factor=0.1 km/h
//   0x51 (81) Vehicle_info_1: Steering_angle_Feedback @ byte5-6, 16bit LE signed, factor=0.1 deg

static void can_reader_thread(const std::string& iface)
{
    int sock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock < 0) {
        std::cerr << "[CAN] socket 생성 실패\n";
        return;
    }

    struct ifreq ifr{};
    std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) {
        std::cerr << "[CAN] " << iface << " 인터페이스 없음\n";
        close(sock);
        return;
    }

    struct sockaddr_can addr{};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[CAN] bind 실패\n";
        close(sock);
        return;
    }

    // 필터: 0x51, 0x52 만 수신
    struct can_filter filters[2];
    filters[0].can_id   = 0x51;
    filters[0].can_mask = CAN_SFF_MASK;
    filters[1].can_id   = 0x52;
    filters[1].can_mask = CAN_SFF_MASK;
    setsockopt(sock, SOL_CAN_RAW, CAN_RAW_FILTER, &filters, sizeof(filters));

    std::cout << "[CAN] " << iface << " 수신 시작 (0x51, 0x52)\n";

    struct can_frame frame;
    while (g_running) {
        int nbytes = read(sock, &frame, sizeof(frame));
        if (nbytes < 0) break;
        if (nbytes < (int)sizeof(frame)) continue;

        if (frame.can_id == 0x52 && frame.can_dlc >= 3) {
            uint16_t raw = (uint16_t)frame.data[1] | ((uint16_t)frame.data[2] << 8);
            g_vehicle_speed_kmh.store(raw * 0.1f, std::memory_order_relaxed);
            g_override_feedback.store(frame.data[0], std::memory_order_relaxed);
        }
        else if (frame.can_id == 0x51 && frame.can_dlc >= 8) {
            int16_t raw = (int16_t)((uint16_t)frame.data[5] | ((uint16_t)frame.data[6] << 8));
            g_steering_angle_deg.store(raw * 0.1f, std::memory_order_relaxed);
            g_switch_state.store(frame.data[7], std::memory_order_relaxed);
        }
    }

    close(sock);
    std::cout << "[CAN] 수신 종료\n";
}

// ── Pure Pursuit 조향 계산 ───────────────────────────────────────────────────
// route_wps (ego: x=전방m, y=좌측m) → 조향각 (deg)
static int g_pp_target_idx = 0;  // PP가 선택한 waypoint index

static float pure_pursuit_steering(const std::vector<Waypoint>& wps, float cur_speed_mps)
{
    float ld_min  = g_pp_ld_min.load(std::memory_order_relaxed);
    float ld_max  = g_pp_ld_max.load(std::memory_order_relaxed);
    float ld_gain = g_pp_ld_gain.load(std::memory_order_relaxed);
    float ld = std::max(ld_min, std::min(cur_speed_mps * ld_gain, ld_max));

    // 룩어헤드 거리에 가장 가까운 waypoint 찾기
    int target_idx = 0;
    float min_diff = 1e9f;
    for (size_t i = 0; i < wps.size(); ++i) {
        float dist = std::sqrt(wps[i].x * wps[i].x + wps[i].y * wps[i].y);
        float diff = std::abs(dist - ld);
        if (diff < min_diff) {
            min_diff = diff;
            target_idx = static_cast<int>(i);
        }
    }

    g_pp_target_idx = target_idx;
    float tx = wps[target_idx].x;  // 전방
    float ty = wps[target_idx].y;  // 좌측
    float actual_ld = std::sqrt(tx * tx + ty * ty);
    if (actual_ld < 0.1f) return 0.0f;

    // Pure Pursuit: wheel_angle = atan2(2 * L * ty, ld^2)
    // wheel_angle → handle_angle = wheel_angle * STEER_RATIO
    // Santafe: left = +, right = -
    // ego: ty>0 = left → need positive handle angle
    // steer_test 확인: left=+, right=-
    // atan2(ty>0) > 0 → 그대로 양수 = 좌회전 ✓
    float wheel_rad = std::atan2(2.0f * WHEELBASE * ty, actual_ld * actual_ld);
    float wheel_deg = wheel_rad * (180.0f / M_PI);
    float handle_deg = wheel_deg * STEER_RATIO;

    return std::max(-520.0f, std::min(520.0f, handle_deg));
}

// ── Stanley 조향 계산 ────────────────────────────────────────────────────────
// heading error + cross-track error at front axle
static float stanley_steering(const std::vector<Waypoint>& wps, float cur_speed_mps)
{
    float k_gain = g_st_k_gain.load(std::memory_order_relaxed);
    float k_soft = g_st_k_soft.load(std::memory_order_relaxed);

    // 1. 가장 가까운 waypoint 찾기 (front axle 기준, ego 원점에서 wheelbase만큼 전방)
    float fa_x = WHEELBASE;  // front axle x in ego
    float fa_y = 0.0f;
    int closest_idx = 0;
    float min_dist = 1e9f;
    for (size_t i = 0; i < wps.size(); ++i) {
        float dx = wps[i].x - fa_x;
        float dy = wps[i].y - fa_y;
        float d = dx * dx + dy * dy;
        if (d < min_dist) {
            min_dist = d;
            closest_idx = static_cast<int>(i);
        }
    }

    // 2. cross-track error: 가까운 점의 lateral offset (y)
    // ego좌표에서 y>0 = 좌측이므로, 경로가 좌측에 있으면 좌회전 필요
    float cte = wps[closest_idx].y - fa_y;

    // 3. heading error: 경로의 접선 방향 vs 차량 방향 (ego에서 차량은 항상 +x 방향)
    int next_idx = std::min(closest_idx + 1, static_cast<int>(wps.size()) - 1);
    int prev_idx = std::max(closest_idx - 1, 0);
    float path_dx = wps[next_idx].x - wps[prev_idx].x;
    float path_dy = wps[next_idx].y - wps[prev_idx].y;
    float heading_err = std::atan2(path_dy, path_dx);  // 차량은 heading=0 in ego

    // 4. Stanley: wheel_angle = heading_err + atan2(k * cte, speed + k_soft)
    // cte>0 (left) → positive → left turn = positive handle ✓
    float cte_term = std::atan2(k_gain * cte, std::abs(cur_speed_mps) + k_soft);
    float wheel_rad = heading_err + cte_term;
    float wheel_deg = wheel_rad * (180.0f / M_PI);
    float handle_deg = wheel_deg * STEER_RATIO;

    return std::max(-520.0f, std::min(520.0f, handle_deg));
}

// ── PID 속도 제어 ────────────────────────────────────────────────────────────
struct SpeedPID {
    float kp = 150.0f;
    float ki = 30.0f;
    float kd = 10.0f;
    float integral = 0.0f;
    float prev_error = 0.0f;
    float dt = 0.05f;  // ~20Hz (추론 주기)

    void compute(float desired_mps, float current_mps, int& accel, int& brake)
    {
        float error = desired_mps - current_mps;
        integral += error * dt;
        integral = std::max(-100.0f, std::min(100.0f, integral));  // anti-windup
        float deriv = (error - prev_error) / dt;
        prev_error = error;

        float output = kp * error + ki * integral + kd * deriv;

        if (output > 0.0f) {
            accel = 650 + static_cast<int>(std::min(output, 150.0f));  // max 800
            brake = 0;
        } else {
            accel = 650;  // idle
            brake = static_cast<int>(std::min(-output * 5.0f, 17000.0f));
        }
    }
};

// ── CAN 송신 스레드 (20ms 주기) ──────────────────────────────────────────────
// Control_CMD (0x150=336): Override[0], Alive_Count[1], Angular_Speed_CMD[5]
// Driving_CMD (0x152=338): Accel[0-1], Brake[2-3], Steer[4-5], Gear[6], Reserved[7]

static void can_sender_thread(const std::string& iface)
{
    int sock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock < 0) {
        std::cerr << "[CAN-TX] socket 생성 실패\n";
        return;
    }

    struct ifreq ifr{};
    std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) {
        std::cerr << "[CAN-TX] " << iface << " 인터페이스 없음\n";
        close(sock);
        return;
    }

    struct sockaddr_can addr{};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[CAN-TX] bind 실패\n";
        close(sock);
        return;
    }

    std::cout << "[CAN-TX] " << iface << " 송신 시작 (20ms 주기)\n";

    uint8_t heartbeat = 0;

    while (g_running) {
        bool active = g_override.load(std::memory_order_relaxed);

        // ── Control_CMD (0x150) ──
        {
            struct can_frame frame{};
            frame.can_id  = 0x150;
            frame.can_dlc = 8;
            frame.data[0] = active ? 1 : 0;       // Override: 1=자율, 0=수동
            frame.data[1] = heartbeat;             // Alive_Count
            frame.data[5] = static_cast<uint8_t>(g_cmd_angular_speed.load(std::memory_order_relaxed)); // Angular_Speed_CMD
            write(sock, &frame, sizeof(frame));
        }

        // ── Driving_CMD (0x152) ──
        if (active) {
            struct can_frame frame{};
            frame.can_id  = 0x152;
            frame.can_dlc = 8;

            int accel = g_cmd_accel.load(std::memory_order_relaxed);
            int brake = g_cmd_brake.load(std::memory_order_relaxed);
            int16_t steer = static_cast<int16_t>(g_cmd_steer_deg.load(std::memory_order_relaxed));
            int gear = g_cmd_gear.load(std::memory_order_relaxed);

            // 속도 제어 비활성화 시 accel=idle, brake=0
            if (g_ctrl_mode != MODE_PP_FULL && g_ctrl_mode != MODE_ST_FULL) {
                accel = 650;
                brake = 0;
            }

            frame.data[0] = accel & 0xFF;
            frame.data[1] = (accel >> 8) & 0xFF;
            frame.data[2] = brake & 0xFF;
            frame.data[3] = (brake >> 8) & 0xFF;
            uint16_t steer_u;
            std::memcpy(&steer_u, &steer, sizeof(steer));
            frame.data[4] = steer_u & 0xFF;
            frame.data[5] = (steer_u >> 8) & 0xFF;
            frame.data[6] = static_cast<uint8_t>(gear);
            frame.data[7] = 0;  // Reserved
            write(sock, &frame, sizeof(frame));
        }

        heartbeat = (heartbeat < 255) ? heartbeat + 1 : 0;

        // 20ms 대기
        usleep(20000);
    }

    // 종료 시 수동 모드로 복귀
    {
        struct can_frame frame{};
        frame.can_id  = 0x150;
        frame.can_dlc = 8;
        frame.data[0] = 0;  // Override off
        frame.data[1] = heartbeat;
        write(sock, &frame, sizeof(frame));
    }

    close(sock);
    std::cout << "[CAN-TX] 송신 종료\n";
}

// ── 터미널 키 입력 스레드 (no-gui 모드용) ────────────────────────────────────
static struct termios g_orig_termios;

static void enable_raw_mode()
{
    tcgetattr(STDIN_FILENO, &g_orig_termios);
    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;  // 100ms timeout
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

static void disable_raw_mode()
{
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
}

static void terminal_key_thread()
{
    enable_raw_mode();
    while (g_running) {
        char c = 0;
        if (read(STDIN_FILENO, &c, 1) == 1) {
            if (c == 27) {  // ESC or arrow
                char seq[2];
                if (read(STDIN_FILENO, &seq[0], 1) == 1 &&
                    read(STDIN_FILENO, &seq[1], 1) == 1) {
                    if (seq[0] == '[') {
                        if (seq[1] == 'D') g_hlc = HLC_LEFT;         // LEFT
                        if (seq[1] == 'C') g_hlc = HLC_RIGHT;        // RIGHT
                        if (seq[1] == 'A') {
                            tune_adjust(g_tune_target, 0.5f);
                            printf("\n[TUNE] %s = %.1f\n", tune_name(g_tune_target), tune_get(g_tune_target));
                        }
                        if (seq[1] == 'B') {
                            tune_adjust(g_tune_target, -0.5f);
                            printf("\n[TUNE] %s = %.1f\n", tune_name(g_tune_target), tune_get(g_tune_target));
                        }
                    }
                } else {
                    g_running = false;  // ESC alone = quit
                }
            }
            else if (c == '1') g_hlc = HLC_LEFT;
            else if (c == '2') g_hlc = HLC_RIGHT;
            else if (c == '3') g_hlc = HLC_STRAIGHT;
            else if (c == '4') g_hlc = HLC_FOLLOW_LANE;
            else if (c == '5') g_hlc = HLC_CHANGE_LEFT;
            else if (c == '6') g_hlc = HLC_CHANGE_RIGHT;
            else if (c == 'o' || c == 'O') {
                bool cur = g_override.load();
                g_override.store(!cur);
                printf("\n[CTRL] Override %s\n", !cur ? "ON (AUTO)" : "OFF (MANUAL)");
            }
            else if (c == 't' || c == 'T') {
                g_tune_target = static_cast<TuneTarget>((g_tune_target + 1) % TUNE_COUNT);
                printf("\n[TUNE] target: %s = %.1f\n", tune_name(g_tune_target), tune_get(g_tune_target));
            }
        }
    }
    disable_raw_mode();
}

static void key_callback(GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
        // HLC
        case GLFW_KEY_1: case GLFW_KEY_LEFT:  g_hlc = HLC_LEFT;        break;
        case GLFW_KEY_2: case GLFW_KEY_RIGHT: g_hlc = HLC_RIGHT;       break;
        case GLFW_KEY_3:                      g_hlc = HLC_STRAIGHT;    break;
        case GLFW_KEY_4: case GLFW_KEY_SPACE: g_hlc = HLC_FOLLOW_LANE; break;
        case GLFW_KEY_5:                      g_hlc = HLC_CHANGE_LEFT;  break;
        case GLFW_KEY_6:                      g_hlc = HLC_CHANGE_RIGHT; break;
        // Override
        case GLFW_KEY_O: {
            bool cur = g_override.load();
            g_override.store(!cur);
            std::cout << "[CTRL] Override " << (!cur ? "ON (AUTO)" : "OFF (MANUAL)") << "\n";
            break;
        }
        // Tuning: T=대상 전환, UP/DOWN=값 조절
        case GLFW_KEY_T: {
            g_tune_target = static_cast<TuneTarget>((g_tune_target + 1) % TUNE_COUNT);
            std::cout << "[TUNE] target: " << tune_name(g_tune_target)
                      << " = " << tune_get(g_tune_target) << "\n";
            break;
        }
        case GLFW_KEY_UP: {
            tune_adjust(g_tune_target, 0.5f);
            std::cout << "[TUNE] " << tune_name(g_tune_target)
                      << " = " << tune_get(g_tune_target) << "\n";
            break;
        }
        case GLFW_KEY_DOWN: {
            tune_adjust(g_tune_target, -0.5f);
            std::cout << "[TUNE] " << tune_name(g_tune_target)
                      << " = " << tune_get(g_tune_target) << "\n";
            break;
        }
        default: break;
    }
}

// ── Waypoint ───────────────────────────────────────────────────────────────────

struct InferenceResult {
    std::vector<Waypoint> speed_wps;
    std::vector<Waypoint> route_wps;
    std::vector<float> lat_logits;      // [4]
    std::vector<float> lon_logits;      // [8]
    std::vector<float> ctx_speed_kind;  // [4]
    std::vector<float> ctx_speed_sub;   // [6]
    std::vector<float> ctx_route_kind;  // [3]
    std::vector<float> ctx_route_sub;   // [5]
};

// ── 이미지 전처리 (RGBA) ──────────────────────────────────────────────────────

static float ftheta_radius_to_theta(float r)
{
    return 0.0010337684070691f * r
         + 2.2048213921266e-08f * r * r
         + 5.6866707523473e-12f * r * r * r
         + 3.1827639075709e-14f * r * r * r * r;
}

static float ftheta_theta_to_radius(float theta, float max_radius)
{
    float lo = 0.0f, hi = max_radius;
    for (int i = 0; i < 24; ++i) {
        float mid = 0.5f * (lo + hi);
        if (ftheta_radius_to_theta(mid) < theta)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5f * (lo + hi);
}

static float sample_rgba_channel(const uint8_t* rgba_data, int src_h, int src_w,
                          size_t src_pitch, float ix, float iy, int c)
{
    if (ix < 0.0f || iy < 0.0f || ix > static_cast<float>(src_w - 1) ||
        iy > static_cast<float>(src_h - 1))
        return 0.0f;

    int x0 = std::max(0, std::min(static_cast<int>(std::floor(ix)), src_w - 1));
    int y0 = std::max(0, std::min(static_cast<int>(std::floor(iy)), src_h - 1));
    int x1 = std::min(x0 + 1, src_w - 1);
    int y1 = std::min(y0 + 1, src_h - 1);
    float dx = ix - static_cast<float>(x0);
    float dy = iy - static_cast<float>(y0);

    const uint8_t* row0 = rgba_data + y0 * src_pitch;
    const uint8_t* row1 = rgba_data + y1 * src_pitch;
    float v00 = row0[x0 * 4 + c];
    float v01 = row0[x1 * 4 + c];
    float v10 = row1[x0 * 4 + c];
    float v11 = row1[x1 * 4 + c];
    return (1.0f - dy) * ((1.0f - dx) * v00 + dx * v01)
         +        dy  * ((1.0f - dx) * v10 + dx * v11);
}

static void rectilinear_to_ftheta_pixel(float rect_u, float rect_v, int src_w, int src_h,
                                 float& ix, float& iy)
{
    const float hfov = SIM_HFOV_DEG * static_cast<float>(M_PI) / 180.0f;
    const float vfov = SIM_VFOV_DEG * static_cast<float>(M_PI) / 180.0f;
    const float fx = src_w / (2.0f * std::tan(hfov * 0.5f));
    const float fy = src_h / (2.0f * std::tan(vfov * 0.5f));
    const float rcx = (src_w - 1) * 0.5f;
    const float rcy = (src_h - 1) * 0.5f;

    const float x = (rect_u - rcx) / fx;
    const float y = (rect_v - rcy) / fy;
    const float rho = std::sqrt(x * x + y * y);
    const float theta = std::atan2(rho, 1.0f);

    const float cx = BONNET_CX * (static_cast<float>(src_w) / BONNET_WIDTH);
    const float cy = BONNET_CY * (static_cast<float>(src_h) / BONNET_HEIGHT);
    const float max_x = std::max(cx, static_cast<float>(src_w) - cx);
    const float max_y = std::max(cy, static_cast<float>(src_h) - cy);
    const float radius = ftheta_theta_to_radius(theta, std::sqrt(max_x * max_x + max_y * max_y));
    const float dir = radius / std::max(rho, 1e-6f);
    ix = (rho < 1e-6f) ? cx : (cx + x * dir);
    iy = (rho < 1e-6f) ? cy : (cy + y * dir);
}

static void preprocess_image_rgba(const uint8_t* rgba_data, int src_h, int src_w,
                           size_t src_pitch, float* out_buf,
                           int camera_view_mode)
{
    // 학습과 동일: 하단 4.8/16 비율 크롭 후 resize
    int crop_h = static_cast<int>(src_h - (src_h * 4.8f) / 16.0f);

    const float scale_y = static_cast<float>(crop_h) / INPUT_H;
    const float scale_x = static_cast<float>(src_w) / INPUT_W;

    for (int c = 0; c < 3; ++c) {
        for (int oh = 0; oh < INPUT_H; ++oh) {
            const float iy = (oh + 0.5f) * scale_y - 0.5f;
            const int   y0 = std::max(static_cast<int>(iy), 0);
            const int   y1 = std::min(y0 + 1, src_h - 1);
            const float dy = iy - static_cast<float>(y0);

            for (int ow = 0; ow < INPUT_W; ++ow) {
                const float ix = (ow + 0.5f) * scale_x - 0.5f;
                const int   x0 = std::max(static_cast<int>(ix), 0);
            const int   x1 = std::min(x0 + 1, src_w - 1);
            const float dx = ix - static_cast<float>(x0);

            const uint8_t* row0 = rgba_data + y0 * src_pitch;
            const uint8_t* row1 = rgba_data + y1 * src_pitch;
                const float v00 = row0[x0 * 4 + c] / 255.0f;
                const float v01 = row0[x1 * 4 + c] / 255.0f;
                const float v10 = row1[x0 * 4 + c] / 255.0f;
                const float v11 = row1[x1 * 4 + c] / 255.0f;
                const float val = (1 - dy) * ((1 - dx) * v00 + dx * v01)
                                +      dy  * ((1 - dx) * v10 + dx * v11);

                out_buf[c * INPUT_H * INPUT_W + oh * INPUT_W + ow] =
                    (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }

    if (camera_view_mode != CAMERA_VIEW_FTHETA_RECTILINEAR)
        return;

    for (int c = 0; c < 3; ++c) {
        for (int oh = 0; oh < INPUT_H; ++oh) {
            const float rect_v = (oh + 0.5f) * scale_y - 0.5f;
            for (int ow = 0; ow < INPUT_W; ++ow) {
                const float rect_u = (ow + 0.5f) * scale_x - 0.5f;
                float src_x, src_y;
                rectilinear_to_ftheta_pixel(rect_u, rect_v, src_w, src_h, src_x, src_y);
                float v = sample_rgba_channel(rgba_data, src_h, src_w, src_pitch, src_x, src_y, c);
                float val = v / 255.0f;
                out_buf[c * INPUT_H * INPUT_W + oh * INPUT_W + ow] =
                    (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
}

// ── TokenDriveInference ────────────────────────────────────────────────────────

class TokenDriveInference {
public:
    explicit TokenDriveInference(const std::string& model_path, bool use_cuda = true)
        : env_(ORT_LOGGING_LEVEL_WARNING, "tokendrive")
    {
        Ort::SessionOptions sess_opts;
        sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sess_opts.SetIntraOpNumThreads(1);

        if (use_cuda) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            sess_opts.AppendExecutionProvider_CUDA(cuda_opts);
            std::cout << "[inference] CUDAExecutionProvider 사용\n";
#else
            std::cout << "[inference] USE_CUDA 미정의, CPU로 실행\n";
#endif
        }

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), sess_opts);

        camera_buf_.assign(1 * 1 * 1 * 3 * INPUT_H * INPUT_W, 0.0f);
        speed_buf_.assign(1, 0.0f);
        hlc_buf_.assign(1, HLC_FOLLOW_LANE);
        speed_wps_buf_.assign(NUM_SPEED_WPS * 2, 0.0f);
        route_wps_buf_.assign(NUM_ROUTE_WPS * 2, 0.0f);
        lat_logits_buf_.assign(4, 0.0f);
        lon_logits_buf_.assign(8, 0.0f);
        ctx_speed_kind_buf_.assign(4, 0.0f);
        ctx_speed_sub_buf_.assign(6, 0.0f);
        ctx_route_kind_buf_.assign(3, 0.0f);
        ctx_route_sub_buf_.assign(5, 0.0f);

        Ort::AllocatorWithDefaultOptions alloc;
        input_name_strs_.reserve(session_->GetInputCount());
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            char* raw = session_->GetInputName(i, alloc);
            input_name_strs_.push_back(raw);
            alloc.Free(raw);
        }
        output_name_strs_.reserve(session_->GetOutputCount());
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            char* raw = session_->GetOutputName(i, alloc);
            output_name_strs_.push_back(raw);
            alloc.Free(raw);
        }
        for (const auto& s : input_name_strs_)  input_names_.push_back(s.c_str());
        for (const auto& s : output_name_strs_) output_names_.push_back(s.c_str());

        std::cout << "[inference] 모델 로드 완료: " << model_path << "\n";
    }

    InferenceResult run()
    {
        Ort::MemoryInfo mem_cpu = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        const std::vector<int64_t> cam_shape = {1, 1, 1, 3, INPUT_H, INPUT_W};
        const std::vector<int64_t> spd_shape = {1, 1};
        const std::vector<int64_t> hlc_shape = {1};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, camera_buf_.data(), camera_buf_.size(),
            cam_shape.data(), cam_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, speed_buf_.data(), speed_buf_.size(),
            spd_shape.data(), spd_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem_cpu, hlc_buf_.data(), hlc_buf_.size(),
            hlc_shape.data(), hlc_shape.size()));

        const std::vector<int64_t> spd_wps_shape       = {1, NUM_SPEED_WPS, 2};
        const std::vector<int64_t> rte_wps_shape       = {1, NUM_ROUTE_WPS, 2};
        const std::vector<int64_t> lat_logits_shape    = {1, 4};
        const std::vector<int64_t> lon_logits_shape    = {1, 8};
        const std::vector<int64_t> ctx_spd_kind_shape  = {1, 4};
        const std::vector<int64_t> ctx_spd_sub_shape   = {1, 6};
        const std::vector<int64_t> ctx_rte_kind_shape  = {1, 3};
        const std::vector<int64_t> ctx_rte_sub_shape   = {1, 5};

        std::vector<Ort::Value> outputs;
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, speed_wps_buf_.data(), speed_wps_buf_.size(),
            spd_wps_shape.data(), spd_wps_shape.size()));
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, route_wps_buf_.data(), route_wps_buf_.size(),
            rte_wps_shape.data(), rte_wps_shape.size()));
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, lat_logits_buf_.data(), lat_logits_buf_.size(),
            lat_logits_shape.data(), lat_logits_shape.size()));
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, lon_logits_buf_.data(), lon_logits_buf_.size(),
            lon_logits_shape.data(), lon_logits_shape.size()));
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, ctx_speed_kind_buf_.data(), ctx_speed_kind_buf_.size(),
            ctx_spd_kind_shape.data(), ctx_spd_kind_shape.size()));
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, ctx_speed_sub_buf_.data(), ctx_speed_sub_buf_.size(),
            ctx_spd_sub_shape.data(), ctx_spd_sub_shape.size()));
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, ctx_route_kind_buf_.data(), ctx_route_kind_buf_.size(),
            ctx_rte_kind_shape.data(), ctx_rte_kind_shape.size()));
        outputs.push_back(Ort::Value::CreateTensor<float>(
            mem_cpu, ctx_route_sub_buf_.data(), ctx_route_sub_buf_.size(),
            ctx_rte_sub_shape.data(), ctx_rte_sub_shape.size()));

        session_->Run(Ort::RunOptions{nullptr},
                      input_names_.data(),  inputs.data(),  inputs.size(),
                      output_names_.data(), outputs.data(), outputs.size());

        InferenceResult result;
        result.speed_wps.resize(NUM_SPEED_WPS);
        result.route_wps.resize(NUM_ROUTE_WPS);
        for (int i = 0; i < NUM_SPEED_WPS; ++i)
            result.speed_wps[i] = {speed_wps_buf_[i*2], speed_wps_buf_[i*2+1]};
        for (int i = 0; i < NUM_ROUTE_WPS; ++i)
            result.route_wps[i] = {route_wps_buf_[i*2], route_wps_buf_[i*2+1]};
        result.lat_logits     = lat_logits_buf_;
        result.lon_logits     = lon_logits_buf_;
        result.ctx_speed_kind = ctx_speed_kind_buf_;
        result.ctx_speed_sub  = ctx_speed_sub_buf_;
        result.ctx_route_kind = ctx_route_kind_buf_;
        result.ctx_route_sub  = ctx_route_sub_buf_;
        return result;
    }

    float*   camera_data() { return camera_buf_.data(); }
    float&   speed_ref()   { return speed_buf_[0]; }
    int64_t& hlc_ref()     { return hlc_buf_[0]; }

#ifdef USE_CUDA
    void preprocess_gpu(const uint8_t* d_rgba, int src_h, int src_w,
                        size_t src_pitch, int camera_view_mode,
                        cudaStream_t stream)
    {
        if (!d_preprocess_buf_) {
            size_t buf_bytes = 3 * INPUT_H * INPUT_W * sizeof(float);
            cudaMalloc(&d_preprocess_buf_, buf_bytes);
        }
        preprocess_image_rgba_cuda(d_rgba, src_h, src_w, src_pitch,
                                   d_preprocess_buf_, INPUT_H, INPUT_W,
                                   4.8f / 16.0f, camera_view_mode, stream);
        cudaMemcpyAsync(camera_buf_.data(), d_preprocess_buf_,
                        camera_buf_.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    ~TokenDriveInference() {
        if (d_preprocess_buf_) cudaFree(d_preprocess_buf_);
    }
#endif

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string>  input_name_strs_, output_name_strs_;
    std::vector<const char*>  input_names_, output_names_;
    std::vector<float>   camera_buf_, speed_buf_, speed_wps_buf_, route_wps_buf_;
    std::vector<float>   lat_logits_buf_, lon_logits_buf_;
    std::vector<float>   ctx_speed_kind_buf_, ctx_speed_sub_buf_;
    std::vector<float>   ctx_route_kind_buf_, ctx_route_sub_buf_;
    std::vector<int64_t> hlc_buf_;
#ifdef USE_CUDA
    float* d_preprocess_buf_ = nullptr;
#endif
};

// ── 유틸 ───────────────────────────────────────────────────────────────────────

float estimate_desired_speed(const std::vector<Waypoint>& speed_wps)
{
    const float dx = speed_wps[2].x - speed_wps[0].x;
    const float dy = speed_wps[2].y - speed_wps[0].y;
    return std::sqrt(dx * dx + dy * dy) * 2.0f;
}

static const char* hlc_name(int64_t hlc) {
    switch (hlc) {
        case HLC_LEFT:         return "LEFT";
        case HLC_RIGHT:        return "RIGHT";
        case HLC_STRAIGHT:     return "STRAIGHT";
        case HLC_FOLLOW_LANE:  return "FOLLOW_LANE";
        case HLC_CHANGE_LEFT:  return "CHANGE_LEFT";
        case HLC_CHANGE_RIGHT: return "CHANGE_RIGHT";
        default: return "UNKNOWN";
    }
}

// ── Ego좌표 → 화면좌표 변환 ──────────────────────────────────────────────────

/**
 * Ego 좌표 waypoint → 화면 좌표 [0, 1] 변환.
 * 화면 좌표계: (0,0)=좌상, (1,1)=우하, X=오른쪽, Y=아래쪽
 * ego: x=전방(m), y=좌측(m)
 * 화면: 하단 중앙이 차량 위치, 위쪽이 전방
 */
static const float BEV_RANGE = 15.0f;  // 고정 범위 15m (waypoint 12.8m + 여유)

void ego_to_screen(float ego_x, float ego_y, float& sx, float& sy)
{
    // 등비율 BEV: 동일한 range로 X/Y 스케일링 → 1m = 1m
    // 화면 하반부(50%) 사용, 차량은 하단 중앙
    const float proj_scale = 0.5f;

    // ego_y(좌측 양수) → 화면 X (좌측 = 작은 값)
    sx = 0.5f - (ego_y / (2.0f * BEV_RANGE)) * proj_scale;
    // ego_x(전방) → 화면 Y (전방 = 위 = 작은 값, 차량 = 아래 = 1.0)
    sy = 1.0f - (ego_x / BEV_RANGE) * proj_scale;
    sx = std::max(0.0f, std::min(1.0f, sx));
    sy = std::max(0.0f, std::min(1.0f, sy));
}

bool waypoint_to_camera_screen(const Waypoint& wp, int img_w, int img_h,
                               float& sx, float& sy)
{
    // Waypoint is ego/rig FLU: x=forward, y=left, z=ground.
    const float rx = wp.x - BONNET_CAM_X;
    const float ry = wp.y - BONNET_CAM_Y;
    const float rz = 0.0f - BONNET_CAM_Z;

    // Display projection: lateral sign is flipped to match the live camera overlay
    const float ox = ry;
    const float oy = -rz;
    const float oz = rx;
    if (oz <= 1e-3f) return false;

    const float rho = std::sqrt(ox * ox + oy * oy);
    const float theta = std::atan2(rho, oz);
    const float cx = BONNET_CX * (static_cast<float>(img_w) / BONNET_WIDTH);
    const float cy = BONNET_CY * (static_cast<float>(img_h) / BONNET_HEIGHT);
    const float max_x = std::max(cx, static_cast<float>(img_w) - cx);
    const float max_y = std::max(cy, static_cast<float>(img_h) - cy);
    const float radius = ftheta_theta_to_radius(theta, std::sqrt(max_x * max_x + max_y * max_y));
    const float dir = radius / std::max(rho, 1e-6f);
    const float u = (rho < 1e-6f) ? cx : (cx + ox * dir);
    const float v = (rho < 1e-6f) ? cy : (cy + oy * dir);
    if (u < 0.0f || u >= img_w || v < 0.0f || v >= img_h)
        return false;

    sx = u / static_cast<float>(img_w);
    sy = v / static_cast<float>(img_h);
    return true;
}

/**
 * Waypoint 목록에서 표시 범위를 자동 계산.
 * 최대 전방/좌우 거리에 여유(1.2배)를 줌.
 */
void compute_wp_range(const std::vector<Waypoint>& wps,
                      float& range_fwd, float& range_lat)
{
    float max_fwd = 0.1f, max_lat = 0.1f;  // 최소값
    for (const auto& wp : wps) {
        max_fwd = std::max(max_fwd, std::abs(wp.x));
        max_lat = std::max(max_lat, std::abs(wp.y));
    }
    range_fwd = max_fwd * 1.2f;
    range_lat = std::max(max_lat * 1.2f, range_fwd * 0.5f);  // 종횡비 보정
}

// ── DW 에러 체크 ──────────────────────────────────────────────────────────────

#define CHECK_DW(call)                                                         \
    do {                                                                       \
        dwStatus _s = (call);                                                  \
        if (_s != DW_SUCCESS) {                                                \
            std::cerr << "[DW ERROR] " << dwGetStatusName(_s) << " at "        \
                      << __FILE__ << ":" << __LINE__ << "\n";                  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static bool is_camera_device_busy()
{
    return std::system("fuser -s /dev/isc-mgr.1.e >/dev/null 2>&1") == 0;
}

static void run_camera_cleanup_command(const char* cmd)
{
    int rc = std::system(cmd);
    (void)rc;
}

static void free_camera_device_if_busy()
{
    if (!is_camera_device_busy())
        return;

    std::cerr << "[camera] /dev/isc-mgr.1.e is busy; terminating camera users...\n";

    const char* term_cmds[] = {
        "fuser -k -TERM /dev/isc-mgr.1.e >/dev/null 2>&1",
        "sudo -n fuser -k -TERM /dev/isc-mgr.1.e >/dev/null 2>&1",
        "pkill -TERM -f /usr/local/driveworks/tools/preFlightChecker >/dev/null 2>&1",
        "sudo -n pkill -TERM -f /usr/local/driveworks/tools/preFlightChecker >/dev/null 2>&1",
        "pkill -TERM -f /usr/local/driveworks/apps/roadrunner >/dev/null 2>&1",
        "sudo -n pkill -TERM -f /usr/local/driveworks/apps/roadrunner >/dev/null 2>&1"
    };
    for (const char* cmd : term_cmds)
        run_camera_cleanup_command(cmd);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    if (!is_camera_device_busy()) {
        std::cerr << "[camera] camera device released\n";
        return;
    }

    std::cerr << "[camera] still busy; sending SIGKILL to camera users...\n";
    const char* kill_cmds[] = {
        "fuser -k -KILL /dev/isc-mgr.1.e >/dev/null 2>&1",
        "sudo -n fuser -k -KILL /dev/isc-mgr.1.e >/dev/null 2>&1"
    };
    for (const char* cmd : kill_cmds)
        run_camera_cleanup_command(cmd);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (is_camera_device_busy())
        std::cerr << "[camera] device is still busy; run with sudo or stop the owner manually\n";
}

// ── main ───────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // ── 커맨드라인 파싱 ────────────────────────────────────────────────────────
    std::string model_path   = "tokendrive_fp32_wp64_op15_v3_token_on.onnx";
    std::string camera_type  = "ar0231-rccb-bae-sf3324";
    std::string camera_group = "c";
    std::string camera_view_name = "ftheta-rectilinear";
    int camera_view_mode = CAMERA_VIEW_FTHETA_RECTILINEAR;
    bool use_cuda = true;
    bool no_gui = false;
    bool save_frames = false;
    bool log_can = false;
    bool free_camera = false;
    std::string can_log_path;
    std::string rtk_ip = "192.168.1.100";
    int rtk_port = 3002;
    float vehicle_speed_mps = 0.0f;
    int64_t hlc_value = HLC_FOLLOW_LANE;
    bool steer_speed_arg = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--model" || a == "-m") && i + 1 < argc)
            model_path = argv[++i];
        else if (a == "--camera-type" && i + 1 < argc)
            camera_type = argv[++i];
        else if (a == "--camera-group" && i + 1 < argc)
            camera_group = argv[++i];
        else if (a == "--camera-view" && i + 1 < argc) {
            camera_view_name = argv[++i];
            if (camera_view_name == "baseline")
                camera_view_mode = CAMERA_VIEW_BASELINE;
            else if (camera_view_name == "ftheta-rectilinear")
                camera_view_mode = CAMERA_VIEW_FTHETA_RECTILINEAR;
            else {
                std::cerr << "[arg] unknown --camera-view: " << camera_view_name
                          << " (use baseline or ftheta-rectilinear)\n";
                return 1;
            }
        }
        else if (a == "--speed" && i + 1 < argc)
            vehicle_speed_mps = std::stof(argv[++i]);
        else if (a == "--hlc" && i + 1 < argc)
            hlc_value = std::stoll(argv[++i]);
        else if (a == "--steer-speed" && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            g_cmd_angular_speed.store(std::max(0, std::min(255, v)));
            steer_speed_arg = true;
        }
        else if (a == "--cpu")
            use_cuda = false;
        else if (a == "--no-gui")
            no_gui = true;
        else if (a == "--save-frames")
            save_frames = true;
        else if (a == "--log-can")
            log_can = true;
        else if (a == "--free-camera")
            free_camera = true;
        else if (a == "--rtk" && i + 1 < argc) {
            std::string arg = argv[++i];
            auto colon = arg.find(':');
            if (colon != std::string::npos) {
                rtk_ip = arg.substr(0, colon);
                rtk_port = std::stoi(arg.substr(colon + 1));
            } else {
                rtk_ip = arg;
                rtk_port = 3002;
            }
        }
        else if (a == "--no-rtk") {
            rtk_ip = "";
            rtk_port = 0;
        }
        else if (a == "--pp-steer")
            g_ctrl_mode = MODE_PP_STEER;
        else if (a == "--pp-full")
            g_ctrl_mode = MODE_PP_FULL;
        else if (a == "--stanley-steer")
            g_ctrl_mode = MODE_ST_STEER;
        else if (a == "--stanley-full")
            g_ctrl_mode = MODE_ST_FULL;
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage: inference_v3 [options]\n"
                "  --model FILE           ONNX model (default: tokendrive_fp32_wp64_op15_v3_token_on.onnx)\n"
                "  --camera-type TYPE     GMSL camera type\n"
                "  --camera-group GRP     camera group a/b/c/d (default: c)\n"
                "  --camera-view MODE     baseline | ftheta-rectilinear (default: ftheta-rectilinear)\n"
                "  --speed MPS            initial speed m/s (default: 0.0)\n"
                "  --hlc VALUE            HLC 1~6 (default: 4=follow_lane)\n"
                "  --steer-speed VALUE    steering speed command 0~255 (default: 50)\n"
                "  --cpu                  CPU inference\n"
                "  --no-gui               headless mode (no visualization)\n"
                "  --save-frames          save raw camera frames + speed to disk\n"
                "  --log-can              CAN+RTK 데이터 20Hz로 CSV 기록 (평가용)\n"
                "  --rtk IP:PORT          NovAtel RTK TCP (기본: 192.168.1.100:3002)\n"
                "  --no-rtk               RTK 비활성화\n"
                "  --free-camera          terminate current camera users and retry sensor init\n"
                "\n"
                "Control modes (default: none = waypoint only):\n"
                "  --pp-steer             Pure Pursuit steering only\n"
                "  --pp-full              Pure Pursuit steering + speed (DANGER!)\n"
                "  --stanley-steer        Stanley steering only\n"
                "  --stanley-full         Stanley steering + speed (DANGER!)\n"
                "\n"
                "  O key: toggle override (default=manual)\n";
            return 0;
        }
    }

    const char* mode_names[] = {"WAYPOINT ONLY", "PP STEER", "PP FULL", "STANLEY STEER", "STANLEY FULL"};
    bool has_control = (g_ctrl_mode != MODE_NONE);
    bool has_speed   = (g_ctrl_mode == MODE_PP_FULL || g_ctrl_mode == MODE_ST_FULL);

    // ── FULL 모드 안전 확인 ───────────────────────────────────────────────────
    if (has_speed) {
        std::cerr << "\n";
        std::cerr << "╔══════════════════════════════════════════════════════╗\n";
        std::cerr << "║  ⚠  경고: 속도 제어 모드 (FULL) 가 선택되었습니다  ║\n";
        std::cerr << "║     차량이 실제로 가속/감속 명령을 받습니다.         ║\n";
        std::cerr << "║     주변 안전을 확인하고 계속하려면 YES 를 입력하세요 ║\n";
        std::cerr << "╚══════════════════════════════════════════════════════╝\n";
        std::cerr << "  확인 입력 (YES / 그 외 취소): ";
        std::string confirm;
        std::getline(std::cin, confirm);
        if (confirm != "YES") {
            std::cout << "[취소] 안전을 위해 종료합니다.\n";
            return 0;
        }
        std::cerr << "  확인되었습니다. 3초 후 시작합니다...\n";
        sleep(3);
    }

    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  Mode: " << mode_names[g_ctrl_mode] << "\n";
    if (has_speed)
        std::cout << "║  [!] SPEED CONTROL ENABLED (max 15km/h)  ║\n";
    if (has_control)
        std::cout << "║  O key: toggle override (default=manual)  ║\n";
    if (save_frames)
        std::cout << "║  [REC] Frame saving enabled               ║\n";
    std::cout << "║  Camera view: " << camera_view_name << "\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    // ── 데이터 수집 디렉토리 생성 ─────────────────────────────────────────────
    std::string save_dir;
    std::string frames_dir;
    std::string can_dir;
    std::string run_ts;
    if (save_frames || log_can) {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
        localtime_r(&t, &tm_buf);
        char ts[64];
        std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
        run_ts = ts;

        const std::string root_dir = SAVE_ROOT_DIR;
        save_dir = root_dir + "/run_" + run_ts;
        frames_dir = save_dir + "/frames";
        can_dir = save_dir + "/can";

        if (!ensure_directory(root_dir) ||
            !ensure_directory(save_dir) ||
            !ensure_directory(frames_dir) ||
            !ensure_directory(can_dir)) {
            std::cerr << "[save] abort: cannot prepare save route under " << root_dir << "\n";
            return 1;
        }
        if ((save_frames && !verify_directory_writable(frames_dir)) ||
            (log_can && !verify_directory_writable(can_dir))) {
            std::cerr << "[save] abort: save route is not writable\n";
            return 1;
        }

        std::cout << "[save] Root dir verified: " << root_dir << "\n";
        std::cout << "[save] Collection dir: " << save_dir << "\n";
        if (save_frames)
            std::cout << "[save] Frames dir verified: " << frames_dir << "\n";
        if (log_can)
            std::cout << "[save] CAN log dir verified: " << can_dir << "\n";
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. GLFW 윈도우 + EGL 초기화
    // ═══════════════════════════════════════════════════════════════════════════
    GLFWwindow* window = nullptr;
    EGLDisplay eglDisplay = nullptr;
    g_hlc = hlc_value;

    if (!no_gui) {
        std::cout << "[init] GLFW 윈도우 초기화...\n";

        if (!glfwInit()) {
            std::cerr << "[ERROR] glfwInit 실패\n";
            return 1;
        }

        glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

        window = glfwCreateWindow(WIN_W, WIN_H,
                                  "TokenDrive Inference", nullptr, nullptr);
        if (!window) {
            std::cerr << "[ERROR] GLFW 윈도우 생성 실패\n";
            glfwTerminate();
            return 1;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);  // vsync off
        glfwSetKeyCallback(window, key_callback);
        eglDisplay = glfwGetEGLDisplay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. DriveWorks 초기화
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "[init] DriveWorks 초기화...\n";

    dwContextHandle_t dw_ctx = DW_NULL_HANDLE;
    dwSALHandle_t     dw_sal = DW_NULL_HANDLE;
    dwSensorHandle_t  dw_camera = DW_NULL_HANDLE;
    dwImageStreamerHandle_t cuda2cpu_streamer = DW_NULL_HANDLE;
    dwImageStreamerHandle_t cuda2gl_streamer  = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t dw_viz = DW_NULL_HANDLE;
    dwRendererHandle_t dw_renderer = DW_NULL_HANDLE;
    cudaStream_t cuda_stream = nullptr;

    // DW context
    dwContextParameters ctx_params = {};
    ctx_params.eglDisplay = eglDisplay;
    CHECK_DW(dwInitialize(&dw_ctx, DW_VERSION, &ctx_params));

    // Visualization + Renderer (GUI only)
    if (!no_gui) {
        CHECK_DW(dwVisualizationInitialize(&dw_viz, dw_ctx));
        CHECK_DW(dwRenderer_initialize(&dw_renderer, dw_viz));
        dwRect renderRect = {0, 0, static_cast<uint32_t>(WIN_W), static_cast<uint32_t>(WIN_H)};
        dwRenderer_setRect(renderRect, dw_renderer);
    }

    // SAL + 카메라
    CHECK_DW(dwSAL_initialize(&dw_sal, dw_ctx));

    std::string cam_params = "output-format=yuv,fifo-size=3"
                             ",camera-type=" + camera_type +
                             ",camera-group=" + camera_group;
    dwSensorParams sensor_params{};
    sensor_params.protocol   = "camera.gmsl";
    sensor_params.parameters = cam_params.c_str();
    if (free_camera)
        free_camera_device_if_busy();

    dwStatus sensor_status = dwSAL_createSensor(&dw_camera, sensor_params, dw_sal);
    if (sensor_status != DW_SUCCESS && free_camera) {
        std::cerr << "[camera] dwSAL_createSensor failed with "
                  << dwGetStatusName(sensor_status)
                  << "; freeing camera users and retrying...\n";
        free_camera_device_if_busy();
        sensor_status = dwSAL_createSensor(&dw_camera, sensor_params, dw_sal);
    }
    if (sensor_status != DW_SUCCESS) {
        std::cerr << "[DW ERROR] " << dwGetStatusName(sensor_status) << " at "
                  << __FILE__ << ":" << __LINE__ << "\n";
        return 1;
    }

    dwCameraProperties cam_props{};
    CHECK_DW(dwSensorCamera_getSensorProperties(&cam_props, dw_camera));
    std::cout << "[camera] " << cam_props.resolution.x << "x"
              << cam_props.resolution.y << " @ " << cam_props.framerate << " FPS\n";

    // CUDA→CPU 스트리머 (전처리용)
    dwImageProperties cuda_img_props{};
    cuda_img_props.width  = cam_props.resolution.x;
    cuda_img_props.height = cam_props.resolution.y;
    cuda_img_props.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    cuda_img_props.type   = DW_IMAGE_CUDA;
    CHECK_DW(dwImageStreamer_initialize(&cuda2cpu_streamer, &cuda_img_props, DW_IMAGE_CPU, dw_ctx));

    // CUDA→GL 스트리머 (화면 렌더링용, GUI only)
    if (!no_gui)
        CHECK_DW(dwImageStreamerGL_initialize(&cuda2gl_streamer, &cuda_img_props, DW_IMAGE_GL, dw_ctx));

    // CUDA 스트림
#ifdef USE_CUDA
    cudaStreamCreate(&cuda_stream);
    dwSensorCamera_setCUDAStream(cuda_stream, dw_camera);
#endif

    // 카메라 시작
    CHECK_DW(dwSensor_start(dw_camera));
    std::cout << "[camera] 시작 대기...\n";
    {
        dwCameraFrameHandle_t test_frame = DW_NULL_HANDLE;
        dwStatus status = DW_NOT_READY;
        while (status == DW_NOT_READY)
            status = dwSensorCamera_readFrame(&test_frame, 0, 500000, dw_camera);
        if (status != DW_SUCCESS) {
            std::cerr << "[ERROR] 카메라 시작 실패\n"; return 1;
        }
        dwSensorCamera_returnFrame(&test_frame);
    }
    std::cout << "[camera] 준비 완료\n";

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. ONNX 모델 로드
    // ═══════════════════════════════════════════════════════════════════════════
    std::cout << "[init] ONNX 모델 로드...\n";
    TokenDriveInference infer(model_path, use_cuda);
    infer.speed_ref() = vehicle_speed_mps;

    // 튜닝 파라미터 복원
    {
        std::ifstream tf("tune_defaults.cfg");
        if (tf.is_open()) {
            std::string line;
            while (std::getline(tf, line)) {
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string key = line.substr(0, eq);
                float val = std::stof(line.substr(eq + 1));
                if (key == "pp_ld_min")  g_pp_ld_min.store(val);
                else if (key == "pp_ld_max")  g_pp_ld_max.store(val);
                else if (key == "pp_ld_gain") g_pp_ld_gain.store(val);
                else if (key == "st_k_gain")  g_st_k_gain.store(val);
                else if (key == "st_k_soft")  g_st_k_soft.store(val);
                else if (key == "steer_speed" && !steer_speed_arg) g_cmd_angular_speed.store(std::max(0, std::min(255, static_cast<int>(val))));
            }
            std::cout << "[tune] 파라미터 로드: tune_defaults.cfg"
                      << " (ld=" << g_pp_ld_min.load() << "/" << g_pp_ld_max.load() << "/" << g_pp_ld_gain.load()
                      << ", st=" << g_st_k_gain.load() << "/" << g_st_k_soft.load()
                      << ", steer_speed=" << g_cmd_angular_speed.load() << ")\n";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3.5 CAN 스레드 시작
    // ═══════════════════════════════════════════════════════════════════════════
    std::thread can_thread(can_reader_thread, "can0");
    can_thread.detach();

    // RTK GPS 스레드
    if (!rtk_ip.empty()) {
        std::thread rtk_thread(rtk_tcp_thread, rtk_ip, rtk_port);
        rtk_thread.detach();
    }

    // CAN 로거 스레드
    if (log_can) {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
        localtime_r(&t, &tm_buf);
        char ts[64];
        std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
        const std::string log_ts = run_ts.empty() ? std::string(ts) : run_ts;
        can_log_path = can_dir.empty()
            ? (std::string("can_log_") + log_ts + ".csv")
            : (can_dir + "/can_log_" + log_ts + ".csv");
        g_log_running.store(true);
        std::thread log_thread(can_logger_thread, can_log_path);
        log_thread.detach();
    }

    // CAN 송신 스레드 (제어 모드 시)
    if (has_control) {
        std::thread sender(can_sender_thread, "can0");
        sender.detach();
    }

    SpeedPID speed_pid;

    // no-gui: 터미널 키 입력 스레드
    if (no_gui) {
        std::thread key_thread(terminal_key_thread);
        key_thread.detach();
    }

    std::cout << "[init] 초기화 완료\n";
    std::cout << "  키보드 HLC: 1/←=LEFT  2/→=RIGHT  3=STRAIGHT  4/Space=FOLLOW_LANE  5=CHANGE_L  6=CHANGE_R\n";
    std::cout << "  CAN: can0에서 Vehicle_Speed 자동 수신\n";
    if (has_control)
        std::cout << "  제어: O=override, T=tune target, ↑↓=adjust (Steer speed 포함)\n"
                  << "  Steer speed 초기값: " << g_cmd_angular_speed.load() << " (0~255)\n";
    std::cout << "  ESC 또는 Ctrl+C로 종료\n\n";

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. 메인 루프
    // ═══════════════════════════════════════════════════════════════════════════
    int frame_count = 0;
    double total_infer_ms = 0.0;
    int last_img_w = static_cast<int>(BONNET_WIDTH);
    int last_img_h = static_cast<int>(BONNET_HEIGHT);
    using Clock = std::chrono::high_resolution_clock;

    while (g_running && (no_gui || !glfwWindowShouldClose(window)))
    {
        int cur_w = WIN_W, cur_h = WIN_H;
        if (!no_gui) {
            glfwPollEvents();
            glfwGetFramebufferSize(window, &cur_w, &cur_h);
        }

        // ── 카메라 프레임 읽기 (FIFO 드레인 → 최신 프레임만 사용) ───────────────
        // 추론이 카메라보다 느려 FIFO에 쌓인 오래된 프레임을 버리고 최신 프레임 획득
        dwCameraFrameHandle_t frame = DW_NULL_HANDLE;
        {
            dwCameraFrameHandle_t tmp = DW_NULL_HANDLE;
            // timeout=0: non-blocking으로 FIFO에 쌓인 프레임 모두 drain
            while (dwSensorCamera_readFrame(&tmp, 0, 0, dw_camera) == DW_SUCCESS) {
                if (frame != DW_NULL_HANDLE)
                    dwSensorCamera_returnFrame(&frame);
                frame = tmp;
            }
            // FIFO가 비었으면(첫 프레임 또는 타이밍이 맞은 경우) blocking으로 대기
            if (frame == DW_NULL_HANDLE)
                dwSensorCamera_readFrame(&frame, 0, 500000, dw_camera);
        }
        dwStatus status = (frame != DW_NULL_HANDLE) ? DW_SUCCESS : DW_TIME_OUT;
        if (status == DW_END_OF_STREAM) break;
        if (status == DW_TIME_OUT) continue;
        if (status != DW_SUCCESS) {
            std::cerr << "[camera] readFrame: " << dwGetStatusName(status) << "\n";
            break;
        }

        // ── CUDA RGBA 이미지 ──────────────────────────────────────────────────
        dwImageHandle_t img_cuda = DW_NULL_HANDLE;
        status = dwSensorCamera_getImage(&img_cuda, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame);
        if (status != DW_SUCCESS) {
            dwSensorCamera_returnFrame(&frame);
            continue;
        }

        // ── 전처리: CUDA→CPU→preprocess ───────────────────────────────────────
        // ── 전처리: GPU에서 직접 crop+resize+normalize ──────────────────────
        {
            dwImageCUDA* img_cuda_ptr = nullptr;
            dwImage_getCUDA(&img_cuda_ptr, img_cuda);
            last_img_w = static_cast<int>(img_cuda_ptr->prop.width);
            last_img_h = static_cast<int>(img_cuda_ptr->prop.height);
#ifdef USE_CUDA
            infer.preprocess_gpu(
                static_cast<const uint8_t*>(img_cuda_ptr->dptr[0]),
                static_cast<int>(img_cuda_ptr->prop.height),
                static_cast<int>(img_cuda_ptr->prop.width),
                img_cuda_ptr->pitch[0],
                camera_view_mode,
                cuda_stream);
#else
            dwImageStreamer_producerSend(img_cuda, cuda2cpu_streamer);
            dwImageHandle_t img_cpu_handle = DW_NULL_HANDLE;
            status = dwImageStreamer_consumerReceive(&img_cpu_handle, 500000, cuda2cpu_streamer);
            if (status == DW_SUCCESS) {
                dwImageCPU* img_cpu = nullptr;
                dwImage_getCPU(&img_cpu, img_cpu_handle);
                preprocess_image_rgba(
                    img_cpu->data[0],
                    static_cast<int>(img_cpu->prop.height),
                    static_cast<int>(img_cpu->prop.width),
                    img_cpu->pitch[0],
                    infer.camera_data(),
                    camera_view_mode);
                dwImageStreamer_consumerReturn(&img_cpu_handle, cuda2cpu_streamer);
            }
            dwImageStreamer_producerReturn(nullptr, 500000, cuda2cpu_streamer);
#endif
        }

        // ── 프레임 저장 (PPM) — save-frames 시에만 CPU 복사 ────────────────
        if (save_frames) {
            dwImageStreamer_producerSend(img_cuda, cuda2cpu_streamer);
            dwImageHandle_t img_cpu_handle = DW_NULL_HANDLE;
            status = dwImageStreamer_consumerReceive(&img_cpu_handle, 500000, cuda2cpu_streamer);
            if (status == DW_SUCCESS) {
                dwImageCPU* img_cpu = nullptr;
                dwImage_getCPU(&img_cpu, img_cpu_handle);
                const auto frame_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                float spd = g_vehicle_speed_kmh.load(std::memory_order_relaxed);
                char fname[512];
                snprintf(fname, sizeof(fname), "%s/%06d_ts%lld_spd%.1f.ppm",
                         frames_dir.c_str(), frame_count, (long long)frame_ts_us, spd);
                int w = static_cast<int>(img_cpu->prop.width);
                int h = static_cast<int>(img_cpu->prop.height);
                size_t pitch = img_cpu->pitch[0];
                std::ofstream ofs(fname, std::ios::binary);
                if (ofs.is_open()) {
                    ofs << "P6\n" << w << " " << h << "\n255\n";
                    std::vector<uint8_t> rgb_row(w * 3);
                    for (int row = 0; row < h; ++row) {
                        const uint8_t* src = img_cpu->data[0] + row * pitch;
                        for (int col = 0; col < w; ++col) {
                            rgb_row[col * 3 + 0] = src[col * 4 + 0];
                            rgb_row[col * 3 + 1] = src[col * 4 + 1];
                            rgb_row[col * 3 + 2] = src[col * 4 + 2];
                        }
                        ofs.write(reinterpret_cast<const char*>(rgb_row.data()), rgb_row.size());
                    }

                    static std::ofstream frame_log;
                    static bool frame_log_header_written = false;
                    if (!frame_log.is_open())
                        frame_log.open(frames_dir + "/frames.csv", std::ios::out);
                    if (frame_log.is_open()) {
                        if (!frame_log_header_written) {
                            frame_log << "frame_idx,timestamp_us,image_path,speed_kmh,"
                                      << "rtk_lat,rtk_lon,rtk_quality\n";
                            frame_log_header_written = true;
                        }
                        frame_log << frame_count << ","
                                  << frame_ts_us << ","
                                  << fname << ","
                                  << spd << ","
                                  << std::fixed << std::setprecision(8)
                                  << g_rtk_lat.load() << ","
                                  << g_rtk_lon.load() << ","
                                  << g_rtk_quality.load() << "\n";
                        frame_log.flush();
                    }
                }
                dwImageStreamer_consumerReturn(&img_cpu_handle, cuda2cpu_streamer);
            }
            dwImageStreamer_producerReturn(nullptr, 500000, cuda2cpu_streamer);
        }

        // ── CAN 속도 + HLC 키보드 입력 반영 ──────────────────────────────────
        float speed_kmh = g_vehicle_speed_kmh.load(std::memory_order_relaxed);
        infer.speed_ref() = speed_kmh / 3.6f;  // km/h → m/s
        infer.hlc_ref() = g_hlc;

        // ── 추론 직전: FIFO에 새 프레임이 있으면 전처리 갱신 ─────────────────
        {
            dwCameraFrameHandle_t fresh = DW_NULL_HANDLE;
            dwCameraFrameHandle_t tmp = DW_NULL_HANDLE;
            while (dwSensorCamera_readFrame(&tmp, 0, 0, dw_camera) == DW_SUCCESS) {
                if (fresh != DW_NULL_HANDLE)
                    dwSensorCamera_returnFrame(&fresh);
                fresh = tmp;
            }
            if (fresh != DW_NULL_HANDLE) {
                dwSensorCamera_returnFrame(&frame);
                frame = fresh;
                dwImageHandle_t fresh_cuda = DW_NULL_HANDLE;
                if (dwSensorCamera_getImage(&fresh_cuda, DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8, frame) == DW_SUCCESS) {
                    img_cuda = fresh_cuda;
                    dwImageCUDA* ptr = nullptr;
                    dwImage_getCUDA(&ptr, img_cuda);
                    last_img_w = static_cast<int>(ptr->prop.width);
                    last_img_h = static_cast<int>(ptr->prop.height);
#ifdef USE_CUDA
                    infer.preprocess_gpu(
                        static_cast<const uint8_t*>(ptr->dptr[0]),
                        static_cast<int>(ptr->prop.height),
                        static_cast<int>(ptr->prop.width),
                        ptr->pitch[0], camera_view_mode, cuda_stream);
#endif
                }
            }
        }

        // ── 추론 ──────────────────────────────────────────────────────────────
        auto t0 = Clock::now();
        InferenceResult result = infer.run();
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        total_infer_ms += ms;
        frame_count++;
        float desired_speed = std::min(estimate_desired_speed(result.speed_wps),
                                       SPEED_CAP_KMH / 3.6f);

        // ── 제어값 계산 ──────────────────────────────────────────────────────
        float ctrl_steer = 0.0f;
        int   ctrl_accel = 650, ctrl_brake = 0;
        if (has_control) {
            float cur_speed_mps = speed_kmh / 3.6f;

            // 조향: Pure Pursuit or Stanley
            if (g_ctrl_mode == MODE_PP_STEER || g_ctrl_mode == MODE_PP_FULL)
                ctrl_steer = pure_pursuit_steering(result.route_wps, cur_speed_mps);
            else
                ctrl_steer = stanley_steering(result.route_wps, cur_speed_mps);
            g_cmd_steer_deg.store(ctrl_steer, std::memory_order_relaxed);

            // 속도: full 모드만
            if (has_speed) {
                speed_pid.compute(desired_speed, cur_speed_mps, ctrl_accel, ctrl_brake);
                // 안전장치: 히스테리시스 적용 (15 km/h 초과 시 제동, 13 km/h 미만 시 해제)
                static bool safety_brake_active = false;
                if (speed_kmh > SAFETY_BRAKE_ON_KMH)
                    safety_brake_active = true;
                else if (speed_kmh < SAFETY_BRAKE_OFF_KMH)
                    safety_brake_active = false;
                if (safety_brake_active) {
                    ctrl_accel = 650;
                    ctrl_brake = SAFETY_BRAKE_CMD;
                }
                g_cmd_accel.store(ctrl_accel, std::memory_order_relaxed);
                g_cmd_brake.store(ctrl_brake, std::memory_order_relaxed);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // 렌더링 (GUI only)
        // ══════════════════════════════════════════════════════════════════════
        if (!no_gui) {

        glViewport(0, 0, cur_w, cur_h);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        dwRect rect = {0, 0, cur_w, cur_h};
        dwRenderer_setRect(rect, dw_renderer);

        // ── (a) 카메라 이미지 배경 ────────────────────────────────────────────
        dwImageStreamerGL_producerSend(img_cuda, cuda2gl_streamer);
        dwImageHandle_t img_gl = DW_NULL_HANDLE;
        status = dwImageStreamerGL_consumerReceive(&img_gl, 33000, cuda2gl_streamer);
        if (status == DW_SUCCESS) {
            dwImageGL* imageGL = nullptr;
            dwImage_getGL(&imageGL, img_gl);
            dwRenderer_renderTexture(imageGL->tex, imageGL->target, dw_renderer);
            dwImageStreamerGL_consumerReturn(&img_gl, cuda2gl_streamer);
        }
        dwImageStreamerGL_producerReturn(nullptr, 33000, cuda2gl_streamer);

        // depth test 비활성화 (텍스처 위에 2D 오버레이가 보이도록)
        glDisable(GL_DEPTH_TEST);



        // ── (c) Route Waypoints 오버레이 (녹색 선) ───────────────────────────
        {
            std::vector<dwVector2f> wp_ndc;
            wp_ndc.reserve(NUM_ROUTE_WPS + 1);
            wp_ndc.push_back({0.5f, 1.0f});
            for (int i = 0; i < NUM_ROUTE_WPS; ++i) {
                float nx, ny;
                if (waypoint_to_camera_screen(result.route_wps[i], last_img_w, last_img_h, nx, ny))
                    wp_ndc.push_back({nx, ny});
            }

            dwRenderer_setColor(DW_RENDERER_COLOR_RED, dw_renderer);
            dwRenderer_setLineWidth(3.0f, dw_renderer);
            if (wp_ndc.size() >= 2)
                dwRenderer_renderData2D(wp_ndc.data(),
                                        wp_ndc.size(),
                                        DW_RENDER_PRIM_LINESTRIP, dw_renderer);

            dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, dw_renderer);
            dwRenderer_setPointSize(10.0f, dw_renderer);
            std::vector<dwVector2f> wp_markers;
            for (int i = 0; i < NUM_ROUTE_WPS; i += 5) {
                float nx, ny;
                if (waypoint_to_camera_screen(result.route_wps[i], last_img_w, last_img_h, nx, ny))
                    wp_markers.push_back({nx, ny});
            }
            if (!wp_markers.empty())
                dwRenderer_renderData2D(wp_markers.data(), wp_markers.size(),
                                        DW_RENDER_PRIM_POINTLIST, dw_renderer);

            // PP target point (cyan, large)
            if (has_control && g_pp_target_idx >= 0 && g_pp_target_idx < NUM_ROUTE_WPS) {
                float nx, ny;
                if (waypoint_to_camera_screen(result.route_wps[g_pp_target_idx], last_img_w, last_img_h, nx, ny)) {
                    dwVector2f pp_pt = {nx, ny};
                    dwRenderer_setColor({0.0f, 1.0f, 1.0f, 1.0f}, dw_renderer);  // cyan
                    dwRenderer_setPointSize(20.0f, dw_renderer);
                    dwRenderer_renderData2D(&pp_pt, 1, DW_RENDER_PRIM_POINTLIST, dw_renderer);
                }
            }
        }

        // ── (d) Speed Waypoints (빨간 선) ────────────────────────────────────
        {
            std::vector<dwVector2f> spd_ndc;
            spd_ndc.reserve(NUM_SPEED_WPS + 1);
            spd_ndc.push_back({0.5f, 1.0f});
            for (int i = 0; i < NUM_SPEED_WPS; ++i) {
                float nx, ny;
                if (waypoint_to_camera_screen(result.speed_wps[i], last_img_w, last_img_h, nx, ny))
                    spd_ndc.push_back({nx, ny});
            }
            dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, dw_renderer);
            dwRenderer_setLineWidth(3.0f, dw_renderer);
            if (spd_ndc.size() >= 2)
                dwRenderer_renderData2D(spd_ndc.data(),
                                        spd_ndc.size(),
                                        DW_RENDER_PRIM_LINESTRIP, dw_renderer);
        }

        // ── (e) 차량 위치 마커 (하단 중앙) ───────────────────────────────────
        {
            dwVector2f car_pos = {0.5f, 1.0f};  // 하단 중앙
            dwRenderer_setColor(DW_RENDERER_COLOR_WHITE, dw_renderer);
            dwRenderer_setPointSize(15.0f, dw_renderer);
            dwRenderer_renderData2D(&car_pos, 1, DW_RENDER_PRIM_POINTLIST, dw_renderer);
        }

        glEnable(GL_DEPTH_TEST);

        // ── (e) 텍스트 OSD ───────────────────────────────────────────────────
        {
            // OSD 반투명 배경 (좌상단 30% x 30%)
            {
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                float bg_w = 0.45f;  // 화면 가로 45%
                float bg_h = 0.3f;   // 화면 세로 30%
                // dwRenderer [0,1] 좌표, (0,0)=좌상
                // 삼각형 2개로 사각형 채우기
                dwRenderer_setColor({0.0f, 0.0f, 0.0f, 0.5f}, dw_renderer);
                dwVector2f bg_verts[6] = {
                    {0.0f, 0.0f}, {bg_w, 0.0f}, {bg_w, bg_h},  // 삼각형 1
                    {0.0f, 0.0f}, {bg_w, bg_h},  {0.0f, bg_h},  // 삼각형 2
                };
                dwRenderer_renderData2D(bg_verts, 6, DW_RENDER_PRIM_TRIANGLELIST, dw_renderer);
                glDisable(GL_BLEND);
            }

            dwRenderer_setFont(DW_RENDER_FONT_VERDANA_20, dw_renderer);
            dwRenderer_setColor(DW_RENDERER_COLOR_WHITE, dw_renderer);

            char buf[256];
            double avg_ms = (frame_count > 0) ? total_infer_ms / frame_count : 0.0;

            snprintf(buf, sizeof(buf), "Inference: %.1f ms (%.1f FPS)", ms, 1000.0 / ms);
            dwRenderer_renderText(10, cur_h - 30, buf, dw_renderer);

            snprintf(buf, sizeof(buf), "Avg: %.1f ms (%.1f FPS)", avg_ms, 1000.0 / avg_ms);
            dwRenderer_renderText(10, cur_h - 55, buf, dw_renderer);

            snprintf(buf, sizeof(buf), "Target Speed: %.1f m/s (%.1f km/h)",
                     desired_speed, desired_speed * 3.6f);
            dwRenderer_renderText(10, cur_h - 80, buf, dw_renderer);

            snprintf(buf, sizeof(buf), "CAN Speed: %.1f km/h (%.1f m/s) | Steer: %.1f deg",
                     speed_kmh, speed_kmh / 3.6f,
                     g_steering_angle_deg.load(std::memory_order_relaxed));
            dwRenderer_renderText(10, cur_h - 105, buf, dw_renderer);

            snprintf(buf, sizeof(buf), "HLC: %s [1:L 2:R 3:S 4:F]", hlc_name(g_hlc));
            dwRenderer_renderText(10, cur_h - 130, buf, dw_renderer);

            snprintf(buf, sizeof(buf), "Mode: %s", mode_names[g_ctrl_mode]);
            dwRenderer_renderText(10, cur_h - 155, buf, dw_renderer);

            if (has_control) {
                bool ovr = g_override.load(std::memory_order_relaxed);
                if (ovr)
                    dwRenderer_setColor({1.0f, 0.2f, 0.2f, 1.0f}, dw_renderer);
                snprintf(buf, sizeof(buf), "CTRL [O]: %s | Steer CMD: %.1f deg",
                         ovr ? ">>> AUTO <<<" : "MANUAL",
                         ctrl_steer);
                dwRenderer_renderText(10, cur_h - 180, buf, dw_renderer);

                if (ovr)
                    dwRenderer_setColor(DW_RENDERER_COLOR_WHITE, dw_renderer);

                if (has_speed) {
                    snprintf(buf, sizeof(buf), "Accel: %d  Brake: %d  Gear: D",
                             ctrl_accel, ctrl_brake);
                    dwRenderer_renderText(10, cur_h - 205, buf, dw_renderer);
                }
            }

            int next_y = has_speed ? 230 : (has_control ? 205 : 180);

            if (has_control) {
                dwRenderer_setColor({0.5f, 1.0f, 0.5f, 1.0f}, dw_renderer);
                snprintf(buf, sizeof(buf), "TUNE [T/Up/Dn]: %s = %.1f",
                         tune_name(g_tune_target), tune_get(g_tune_target));
                dwRenderer_renderText(10, cur_h - next_y, buf, dw_renderer);
                dwRenderer_setColor(DW_RENDERER_COLOR_WHITE, dw_renderer);
                next_y += 25;
            }

            snprintf(buf, sizeof(buf), "Frame: %d", frame_count);
            dwRenderer_renderText(10, cur_h - next_y, buf, dw_renderer);
            next_y += 25;

            snprintf(buf, sizeof(buf), "Camera view: %s", camera_view_name.c_str());
            dwRenderer_renderText(10, cur_h - next_y, buf, dw_renderer);

            // ── Token OSD (우상단) ────────────────────────────────────────────
            {
                int tx = cur_w - 320;  // 우측 320px
                int ty = cur_h - 20;

                // 반투명 배경
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                dwRenderer_setColor({0.0f, 0.0f, 0.0f, 0.5f}, dw_renderer);
                float bx = static_cast<float>(tx) / cur_w;
                float by_top = 0.0f, by_bot = 0.35f;
                dwVector2f tbg[6] = {
                    {bx, by_top}, {1.0f, by_top}, {1.0f, by_bot},
                    {bx, by_top}, {1.0f, by_bot}, {bx,   by_bot},
                };
                dwRenderer_renderData2D(tbg, 6, DW_RENDER_PRIM_TRIANGLELIST, dw_renderer);
                glDisable(GL_BLEND);

                dwRenderer_setFont(DW_RENDER_FONT_VERDANA_20, dw_renderer);

                auto render_token = [&](const char* name,
                                        const std::vector<float>& logits,
                                        const char** labels, int n_labels,
                                        int& y_pos) {
                    std::vector<float> probs = softmax(logits);
                    int best = argmax(probs);
                    dwRenderer_setColor({0.7f, 0.7f, 0.7f, 1.0f}, dw_renderer);
                    snprintf(buf, sizeof(buf), "%s:", name);
                    dwRenderer_renderText(tx, y_pos, buf, dw_renderer);
                    y_pos -= 20;
                    dwRenderer_setColor({0.3f, 1.0f, 0.3f, 1.0f}, dw_renderer);
                    snprintf(buf, sizeof(buf), "  >> %s (%.2f)", labels[best], probs[best]);
                    dwRenderer_renderText(tx, y_pos, buf, dw_renderer);
                    y_pos -= 22;
                    dwRenderer_setColor(DW_RENDERER_COLOR_WHITE, dw_renderer);
                };

                render_token("lat_action",    result.lat_logits,     LAT_ACTION_LABELS,    4, ty);
                render_token("lon_action",    result.lon_logits,     LON_ACTION_LABELS,    8, ty);
                render_token("spd_kind",      result.ctx_speed_kind, CTX_SPEED_KIND_LABELS,4, ty);
                render_token("spd_sub",       result.ctx_speed_sub,  CTX_SPEED_SUB_LABELS, 6, ty);
                render_token("rte_kind",      result.ctx_route_kind, CTX_ROUTE_KIND_LABELS,3, ty);
                render_token("rte_sub",       result.ctx_route_sub,  CTX_ROUTE_SUB_LABELS, 5, ty);
            }
        }

        // ── 카메라 프레임 렌더링 끝 ─────────────────────────────────────────
            glfwSwapBuffers(window);
        } // end if (!no_gui) rendering

        // ── 카메라 프레임 반환 ────────────────────────────────────────────────
        dwSensorCamera_returnFrame(&frame);

        // 콘솔 출력 (10프레임마다)
        if (frame_count % 10 == 0) {
            float steer_fb = g_steering_angle_deg.load(std::memory_order_relaxed);
            bool ovr = g_override.load(std::memory_order_relaxed);
            const char* lat_pred = LAT_ACTION_LABELS[argmax(result.lat_logits)];
            const char* lon_pred = LON_ACTION_LABELS[argmax(result.lon_logits)];
            const char* spd_kind = CTX_SPEED_KIND_LABELS[argmax(result.ctx_speed_kind)];
            const char* spd_sub  = CTX_SPEED_SUB_LABELS[argmax(result.ctx_speed_sub)];
            const char* rte_kind = CTX_ROUTE_KIND_LABELS[argmax(result.ctx_route_kind)];
            const char* rte_sub  = CTX_ROUTE_SUB_LABELS[argmax(result.ctx_route_sub)];
            printf("\r[F%d] %.1fms  HLC:%s  CAN:%.1fkm/h  SteerFB:%.1f  CMD:%.1f  %s"
                   "  lat:%s  lon:%s  spd:%s/%s  rte:%s/%s   ",
                   frame_count, ms, hlc_name(g_hlc),
                   speed_kmh, steer_fb, ctrl_steer,
                   ovr ? "AUTO" : "MANU",
                   lat_pred, lon_pred, spd_kind, spd_sub, rte_kind, rte_sub);
            fflush(stdout);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 정리
    // ═══════════════════════════════════════════════════════════════════════════
    if (frame_count > 0) {
        double avg_ms = total_infer_ms / frame_count;
        std::cout << "\n[통계] " << frame_count << " 프레임"
                  << "  평균=" << std::fixed << std::setprecision(1) << avg_ms << "ms"
                  << "  FPS=" << (1000.0 / avg_ms) << "\n";
    }

    if (save_frames)
        std::cout << "[save] " << frame_count << " frames saved to: " << frames_dir << "\n";

    // 튜닝 파라미터 변경 시 기본값으로 저장
    {
        float cur_ld_min = g_pp_ld_min.load(), cur_ld_max = g_pp_ld_max.load();
        float cur_ld_gain = g_pp_ld_gain.load();
        float cur_k_gain = g_st_k_gain.load(), cur_k_soft = g_st_k_soft.load();
        int cur_steer_speed = g_cmd_angular_speed.load();
        std::ofstream tf("tune_defaults.cfg");
        if (tf.is_open()) {
            tf << "pp_ld_min=" << cur_ld_min << "\n"
               << "pp_ld_max=" << cur_ld_max << "\n"
               << "pp_ld_gain=" << cur_ld_gain << "\n"
               << "st_k_gain=" << cur_k_gain << "\n"
               << "st_k_soft=" << cur_k_soft << "\n"
               << "steer_speed=" << cur_steer_speed << "\n";
            std::cout << "[tune] 파라미터 저장: tune_defaults.cfg"
                      << " (ld=" << cur_ld_min << "/" << cur_ld_max << "/" << cur_ld_gain
                      << ", st=" << cur_k_gain << "/" << cur_k_soft
                      << ", steer_speed=" << cur_steer_speed << ")\n";
        }
    }

    // CSV 로그 이름 변경
    if (log_can && !can_log_path.empty()) {
        std::cout << "[LOG] 현재 파일: " << can_log_path << "\n";
        std::cout << "[LOG] 이름 변경 (엔터=유지): ";
        std::string new_name;
        std::getline(std::cin, new_name);
        if (!new_name.empty()) {
            if (new_name.find(".csv") == std::string::npos)
                new_name += ".csv";
            std::string new_path = new_name;
            if (new_path.find("/") == std::string::npos && !can_dir.empty())
                new_path = can_dir + "/" + new_path;
            if (std::rename(can_log_path.c_str(), new_path.c_str()) == 0)
                std::cout << "[LOG] 이름 변경: " << new_path << "\n";
            else
                std::cerr << "[LOG] 이름 변경 실패\n";
        }
    }

    std::cout << "[cleanup] 리소스 해제...\n";

    if (dw_renderer) dwRenderer_release(dw_renderer);
    if (cuda2gl_streamer) dwImageStreamerGL_release(cuda2gl_streamer);
    if (cuda2cpu_streamer) dwImageStreamer_release(cuda2cpu_streamer);
    if (dw_viz) dwVisualizationRelease(dw_viz);

    dwSensor_stop(dw_camera);
    dwSAL_releaseSensor(dw_camera);
    dwSAL_release(dw_sal);
    dwRelease(dw_ctx);

#ifdef USE_CUDA
    if (cuda_stream) cudaStreamDestroy(cuda_stream);
#endif

    if (!no_gui) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    std::cout << "[cleanup] 완료\n";
    return 0;
}
