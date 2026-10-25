#include "preprocess_cuda.cuh"

static constexpr int CAMERA_VIEW_FTHETA_RECTILINEAR = 1;

// ImageNet 정규화 상수
__constant__ float c_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float c_std[3]  = {0.229f, 0.224f, 0.225f};

static __device__ float ftheta_radius_to_theta(float r)
{
    return 0.0010337684070691f * r
         + 2.2048213921266e-08f * r * r
         + 5.6866707523473e-12f * r * r * r
         + 3.1827639075709e-14f * r * r * r * r;
}

static __device__ float ftheta_theta_to_radius(float theta, float max_radius)
{
    float lo = 0.0f;
    float hi = max_radius;
    #pragma unroll
    for (int i = 0; i < 18; ++i) {
        float mid = 0.5f * (lo + hi);
        if (ftheta_radius_to_theta(mid) < theta)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5f * (lo + hi);
}

__device__ float bilinear_rgba_channel(
    const uint8_t* __restrict__ src,
    int src_h, int src_w, size_t src_pitch,
    float ix, float iy, int c)
{
    if (ix < 0.0f || iy < 0.0f || ix > static_cast<float>(src_w - 1) || iy > static_cast<float>(src_h - 1))
        return 0.0f;

    int x0 = max(min(static_cast<int>(floorf(ix)), src_w - 1), 0);
    int y0 = max(min(static_cast<int>(floorf(iy)), src_h - 1), 0);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    float dx = ix - static_cast<float>(x0);
    float dy = iy - static_cast<float>(y0);

    const uint8_t* row0 = src + y0 * src_pitch;
    const uint8_t* row1 = src + y1 * src_pitch;
    float v00 = row0[x0 * 4 + c];
    float v01 = row0[x1 * 4 + c];
    float v10 = row1[x0 * 4 + c];
    float v11 = row1[x1 * 4 + c];
    return (1.0f - dy) * ((1.0f - dx) * v00 + dx * v01)
         +        dy  * ((1.0f - dx) * v10 + dx * v11);
}

__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src,
    int src_h, int src_w, size_t src_pitch,
    float* __restrict__ dst,
    int dst_h, int dst_w,
    int crop_h,
    float scale_y, float scale_x)
{
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow >= dst_w || oh >= dst_h) return;

    const float iy = (oh + 0.5f) * scale_y - 0.5f;
    const int y0 = max(static_cast<int>(iy), 0);
    const int y1 = min(y0 + 1, src_h - 1);
    const float dy = iy - static_cast<float>(y0);

    const float ix = (ow + 0.5f) * scale_x - 0.5f;
    const int x0 = max(static_cast<int>(ix), 0);
    const int x1 = min(x0 + 1, src_w - 1);
    const float dx = ix - static_cast<float>(x0);

    const uint8_t* row0 = src + y0 * src_pitch;
    const uint8_t* row1 = src + y1 * src_pitch;

    const float w00 = (1.0f - dy) * (1.0f - dx);
    const float w01 = (1.0f - dy) * dx;
    const float w10 = dy * (1.0f - dx);
    const float w11 = dy * dx;

    const int plane = dst_h * dst_w;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v = w00 * row0[x0 * 4 + c]
                + w01 * row0[x1 * 4 + c]
                + w10 * row1[x0 * 4 + c]
                + w11 * row1[x1 * 4 + c];
        dst[c * plane + oh * dst_w + ow] = (v / 255.0f - c_mean[c]) / c_std[c];
    }
}

__global__ void preprocess_ftheta_rectilinear_kernel(
    const uint8_t* __restrict__ src,
    int src_h, int src_w, size_t src_pitch,
    float* __restrict__ dst,
    int dst_h, int dst_w,
    int crop_h,
    float scale_y, float scale_x)
{
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow >= dst_w || oh >= dst_h) return;

    const float rect_u = (ow + 0.5f) * scale_x - 0.5f;
    const float rect_v = (oh + 0.5f) * scale_y - 0.5f;

    const float hfov = 110.0f * 3.14159265358979323846f / 180.0f;
    const float vfov = 71.0753555839f * 3.14159265358979323846f / 180.0f;
    const float fx = src_w / (2.0f * tanf(hfov * 0.5f));
    const float fy = src_h / (2.0f * tanf(vfov * 0.5f));
    const float rcx = (src_w - 1) * 0.5f;
    const float rcy = (src_h - 1) * 0.5f;

    const float x = (rect_u - rcx) / fx;
    const float y = (rect_v - rcy) / fy;
    const float z = 1.0f;
    const float rho = sqrtf(x * x + y * y);
    const float theta = atan2f(rho, z);

    const float cx = 964.875977f * (static_cast<float>(src_w) / 1920.0f);
    const float cy = 645.572449f * (static_cast<float>(src_h) / 1208.0f);
    const float max_x = fmaxf(cx, static_cast<float>(src_w) - cx);
    const float max_y = fmaxf(cy, static_cast<float>(src_h) - cy);
    const float max_radius = sqrtf(max_x * max_x + max_y * max_y);
    const float radius = ftheta_theta_to_radius(theta, max_radius);
    const float dir = radius / fmaxf(rho, 1e-6f);
    const float ix = (rho < 1e-6f) ? cx : (cx + x * dir);
    const float iy = (rho < 1e-6f) ? cy : (cy + y * dir);

    const int plane = dst_h * dst_w;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v = bilinear_rgba_channel(src, src_h, src_w, src_pitch, ix, iy, c);
        dst[c * plane + oh * dst_w + ow] = (v / 255.0f - c_mean[c]) / c_std[c];
    }
}

void preprocess_image_rgba_cuda(
    const uint8_t* d_rgba,
    int src_h, int src_w,
    size_t src_pitch,
    float* d_out,
    int dst_h, int dst_w,
    float crop_ratio,
    int camera_view_mode,
    cudaStream_t stream)
{
    const int crop_h = static_cast<int>(src_h - src_h * crop_ratio);
    const float scale_y = static_cast<float>(crop_h) / dst_h;
    const float scale_x = static_cast<float>(src_w) / dst_w;

    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    if (camera_view_mode == CAMERA_VIEW_FTHETA_RECTILINEAR) {
        preprocess_ftheta_rectilinear_kernel<<<grid, block, 0, stream>>>(
            d_rgba, src_h, src_w, src_pitch,
            d_out, dst_h, dst_w,
            crop_h, scale_y, scale_x);
    } else {
        preprocess_kernel<<<grid, block, 0, stream>>>(
            d_rgba, src_h, src_w, src_pitch,
            d_out, dst_h, dst_w,
            crop_h, scale_y, scale_x);
    }
}
