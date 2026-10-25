#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// GPU에서 RGBA 이미지를 camera-view 변환 + crop + bilinear resize + ImageNet normalize 수행
// 결과는 CHW float 버퍼 (3 x dst_h x dst_w)에 저장
void preprocess_image_rgba_cuda(
    const uint8_t* d_rgba,     // GPU RGBA 소스 이미지
    int src_h, int src_w,      // 소스 해상도
    size_t src_pitch,          // 소스 행 바이트 수
    float* d_out,              // GPU 출력 CHW float 버퍼
    int dst_h, int dst_w,      // 출력 해상도 (448x448)
    float crop_ratio,          // 하단 크롭 비율 (4.8/16.0)
    int camera_view_mode,      // CameraViewMode
    cudaStream_t stream);
