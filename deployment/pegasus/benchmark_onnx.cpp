/**
 * TokenDrive ONNX Latency Benchmark (C++)
 *
 * ONNX Runtime C++ API로 latency 측정.
 * Python 없이 Pegasus (arm64 / CUDA 10.2)에서 직접 실행.
 *
 * 빌드:
 *   mkdir build && cd build
 *   cmake .. -DORT_INCLUDE_DIR=/path/to/ort/include -DORT_LIBRARY=/path/to/libonnxruntime.so
 *   make -j4
 *
 * 실행:
 *   ./benchmark_onnx --model tokendrive_fp32.onnx --device cuda
 *   ./benchmark_onnx --model tokendrive_fp32.onnx --device cpu
 *   ./benchmark_onnx --model a.onnx b.onnx --device cuda --warmup 30 --iters 100
 */

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// ── 커맨드라인 파싱 ────────────────────────────────────────────────────────────
struct Args {
    std::vector<std::string> models;
    std::string device   = "cuda";
    int batch_size       = 1;
    int warmup           = 30;
    int iters            = 100;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.models.push_back(argv[++i]);
            }
        } else if (a == "--device"     && i + 1 < argc) { args.device     = argv[++i]; }
        else if (a == "--batch-size"   && i + 1 < argc) { args.batch_size = std::stoi(argv[++i]); }
        else if (a == "--warmup"       && i + 1 < argc) { args.warmup     = std::stoi(argv[++i]); }
        else if (a == "--iters"        && i + 1 < argc) { args.iters      = std::stoi(argv[++i]); }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage: benchmark_onnx [options]\n"
                "  --model FILE [FILE ...]  ONNX 파일 (복수 지정 가능)\n"
                "  --device cuda|cpu        실행 장치 (default: cuda)\n"
                "  --batch-size N           배치 크기 (default: 1)\n"
                "  --warmup N               warmup 횟수 (default: 30)\n"
                "  --iters N                측정 횟수 (default: 100)\n";
            exit(0);
        }
    }
    if (args.models.empty()) {
        std::cerr << "ERROR: --model 을 지정하세요.\n";
        exit(1);
    }
    return args;
}

// ── 통계 ───────────────────────────────────────────────────────────────────────
struct Stats {
    double mean_ms;
    double std_ms;
    double p50_ms;
    double p95_ms;
    double max_ms;
};

Stats compute_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / times.size());

    size_t n = times.size();
    return {
        mean,
        stddev,
        times[n / 2],
        times[static_cast<size_t>(n * 0.95)],
        times.back(),
    };
}

// ── CUDA event 타이밍 ──────────────────────────────────────────────────────────
#ifdef USE_CUDA
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void record_start() { cudaEventRecord(start); }
    float elapsed_ms()  {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};
#endif

// ── wall-clock 타이밍 ──────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;
inline double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── 입력 텐서 생성 ─────────────────────────────────────────────────────────────
std::vector<Ort::Value> make_inputs(
    Ort::Session& session,
    Ort::MemoryInfo& mem_info,
    int batch_size,
    // CPU 버퍼들 (lifetime 관리용)
    std::vector<std::vector<float>>& float_bufs,
    std::vector<std::vector<int64_t>>& int_bufs
) {
    size_t n_inputs = session.GetInputCount();
    Ort::AllocatorWithDefaultOptions alloc;

    std::vector<Ort::Value> inputs;
    inputs.reserve(n_inputs);
    float_bufs.clear();
    int_bufs.clear();

    for (size_t i = 0; i < n_inputs; ++i) {
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        ONNXTensorElementDataType dtype = tensor_info.GetElementType();

        // dynamic 차원 (batch) → batch_size로 교체
        for (auto& d : shape) {
            if (d <= 0) d = batch_size;
        }

        int64_t numel = 1;
        for (auto d : shape) numel *= d;

        if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int_bufs.push_back(std::vector<int64_t>(numel, 0));
            auto& buf = int_bufs.back();
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info, buf.data(), buf.size(), shape.data(), shape.size()));
        } else {
            // float32 (또는 float16 → float32로 처리)
            float_bufs.push_back(std::vector<float>(numel, 0.0f));
            auto& buf = float_bufs.back();
            // camera_images: 정규화된 dummy 이미지 (0.0)
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info, buf.data(), buf.size(), shape.data(), shape.size()));
        }
    }
    return inputs;
}

// ── 단일 모델 벤치마크 ─────────────────────────────────────────────────────────
void benchmark_one(const std::string& model_path, const Args& args) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  Model  : " << model_path << "\n";
    std::cout << std::string(60, '=') << "\n";

    // ORT 환경
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark");
    Ort::SessionOptions sess_opts;
    sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sess_opts.SetIntraOpNumThreads(1);

    bool use_cuda = (args.device == "cuda");

    if (use_cuda) {
#ifdef USE_CUDA
        std::cerr << "[DBG] Creating OrtCUDAProviderOptions\n" << std::flush;
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        std::cerr << "[DBG] Calling AppendExecutionProvider_CUDA\n" << std::flush;
        sess_opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cerr << "[DBG] CUDA provider appended OK\n" << std::flush;
        std::cout << "  Provider: CUDAExecutionProvider\n";
#else
        std::cout << "  [warn] USE_CUDA 미정의, CPU로 실행\n";
        use_cuda = false;
#endif
    } else {
        std::cout << "  Provider: CPUExecutionProvider\n";
    }

    // 세션 로드
    std::cout << "  Loading model ...\n";
#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    Ort::Session session(env, wpath.c_str(), sess_opts);
#else
    Ort::Session session(env, model_path.c_str(), sess_opts);
#endif

    // 입력/출력 정보 출력
    Ort::AllocatorWithDefaultOptions alloc;
    std::cout << "\n  Inputs:\n";
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        char* raw = session.GetInputName(i, alloc); std::string name(raw); alloc.Free(raw);
        auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "    " << std::left << std::setw(30) << name << "  [";
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j) std::cout << ", ";
            if (shape[j] < 0) std::cout << "batch";
            else               std::cout << shape[j];
        }
        std::cout << "]\n";
    }
    std::cout << "\n  Outputs:\n";
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        char* raw = session.GetOutputName(i, alloc); std::string name(raw); alloc.Free(raw);
        auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "    " << std::left << std::setw(30) << name << "  [";
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j) std::cout << ", ";
            if (shape[j] < 0) std::cout << "batch";
            else               std::cout << shape[j];
        }
        std::cout << "]\n";
    }

    // 입력 텐서 생성
    // CPU 메모리에 입력 텐서 생성 (ORT가 GPU로 자동 복사)
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<std::vector<float>> float_bufs;
    std::vector<std::vector<int64_t>> int_bufs;
    auto input_tensors = make_inputs(session, mem_info, args.batch_size,
                                     float_bufs, int_bufs);

    // 출력 이름 수집 (strings 먼저 확보 후 c_str 포인터 수집 — 재할당 방지)
    std::vector<std::string> input_name_strs, output_name_strs;
    std::vector<const char*> input_names, output_names;
    input_name_strs.reserve(session.GetInputCount());
    output_name_strs.reserve(session.GetOutputCount());
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        char* raw = session.GetInputName(i, alloc);
        input_name_strs.push_back(raw); alloc.Free(raw);
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        char* raw = session.GetOutputName(i, alloc);
        output_name_strs.push_back(raw); alloc.Free(raw);
    }
    for (const auto& s : input_name_strs)  input_names.push_back(s.c_str());
    for (const auto& s : output_name_strs) output_names.push_back(s.c_str());

    // ── Warm-up ──
    std::cout << "\n  Warm-up (" << args.warmup << " iters) ... " << std::flush;
    for (int i = 0; i < args.warmup; ++i) {
        session.Run(Ort::RunOptions{nullptr},
                    input_names.data(), input_tensors.data(), input_tensors.size(),
                    output_names.data(), output_names.size());
    }
#ifdef USE_CUDA
    if (use_cuda) cudaDeviceSynchronize();
#endif
    std::cout << "done\n";

    // ── Measurement ──
    std::cout << "  Measuring (" << args.iters << " iters) ... " << std::flush;
    std::vector<double> times(args.iters);

#ifdef USE_CUDA
    if (use_cuda) {
        CudaTimer timer;
        for (int i = 0; i < args.iters; ++i) {
            timer.record_start();
            session.Run(Ort::RunOptions{nullptr},
                        input_names.data(), input_tensors.data(), input_tensors.size(),
                        output_names.data(), output_names.size());
            times[i] = static_cast<double>(timer.elapsed_ms());
        }
    } else
#endif
    {
        for (int i = 0; i < args.iters; ++i) {
            auto t0 = Clock::now();
            session.Run(Ort::RunOptions{nullptr},
                        input_names.data(), input_tensors.data(), input_tensors.size(),
                        output_names.data(), output_names.size());
            times[i] = elapsed_ms(t0);
        }
    }
    std::cout << "done\n";

    auto s = compute_stats(times);
    std::string method = use_cuda ? "CUDA event" : "wall-clock";

    std::cout << "\n  [" << method << "]\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    Mean   : " << std::setw(8) << s.mean_ms << " ms\n";
    std::cout << "    Std    : " << std::setw(8) << s.std_ms  << " ms\n";
    std::cout << "    p50    : " << std::setw(8) << s.p50_ms  << " ms\n";
    std::cout << "    p95    : " << std::setw(8) << s.p95_ms  << " ms\n";
    std::cout << "    max    : " << std::setw(8) << s.max_ms  << " ms\n";
    std::cout << "    FPS    : " << std::setw(8) << (1000.0 / s.mean_ms) << "\n";
}

// ── main ───────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    std::cout << "Device     : " << args.device     << "\n";
    std::cout << "Batch size : " << args.batch_size << "\n";
    std::cout << "Warmup     : " << args.warmup     << "\n";
    std::cout << "Iters      : " << args.iters      << "\n";

    struct Result { std::string name; double mean_ms; double std_ms; double p95_ms; };
    std::vector<Result> results;

    for (const auto& model_path : args.models) {
        try {
            benchmark_one(model_path, args);
        } catch (const Ort::Exception& e) {
            std::cerr << "\n[ORT ERROR] " << e.what() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "\n[ERROR] " << e.what() << "\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
