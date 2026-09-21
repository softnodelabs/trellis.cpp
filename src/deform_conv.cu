// Modulated deformable conv2d (matches torchvision.ops.deform_conv2d, v2/modulated).
// Fused kernel: one thread per output element, loops over (Cin, K, K), bilinear-samples x at the
// per-location deformed positions with zero boundary. No im2col buffer.
#include "deform_conv.h"
// Portable across CUDA and HIP: on a HIP compile the toolchain defines __HIP__ /
// __HIP_PLATFORM_AMD__, so map the handful of runtime calls used below onto hip*.
// (Mirrors the shim in decimate_qem.cu; the <<<>>> launch works on both.)
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__) || defined(GGML_USE_HIP)
  #include <hip/hip_runtime.h>
  #define cudaSetDevice           hipSetDevice
  #define cudaMalloc              hipMalloc
  #define cudaFree                hipFree
  #define cudaMemcpy              hipMemcpy
  #define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
  #define cudaDeviceSynchronize   hipDeviceSynchronize
  #define cudaError_t             hipError_t
  #define cudaSuccess             hipSuccess
  #define cudaGetErrorString      hipGetErrorString
#else
  #include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cmath>

namespace trellis {

// Output channels one thread accumulates, and input channels whose weights are staged in shared
// memory at a time.
constexpr int DC_OCT = 32;
constexpr int DC_ICT = 64;
constexpr int DC_THREADS = 128;

// One thread per output pixel and per tile of DC_OCT output channels. For each kernel tap the
// sampling position and its four bilinear corners are worked out once, then reused for every input
// channel; each sample is then reused for every output channel of the tile. The naive form redid
// both per (output channel, input channel, tap), which is Cout times the sampling work.
__global__ void deform_conv_kernel(const float* x, int Cin, int H, int W,
                                   const float* offset, const float* mask,
                                   const float* weight, const float* bias, int Cout, int K, int pad,
                                   float* out) {
    __shared__ float ws[DC_OCT * DC_ICT];                // [o][ic] weights of this tile for one tap
    const long HW = (long)H * W;
    const long pix = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int oc0 = blockIdx.y * DC_OCT;
    const int noc = min(DC_OCT, Cout - oc0);
    const bool live = pix < HW;
    const int oy = live ? (int)(pix / W) : 0, ox = live ? (int)(pix % W) : 0;
    const int K2 = K * K;
    float acc[DC_OCT];
#pragma unroll
    for (int o = 0; o < DC_OCT; ++o) acc[o] = (bias && o < noc) ? bias[oc0 + o] : 0.f;

    for (int t = 0; t < K2; ++t) {
        // The tap's sample: corner offsets and weights, zero weight for a corner outside the map.
        long i1 = 0, i2 = 0, i3 = 0, i4 = 0;
        float w1 = 0.f, w2 = 0.f, w3 = 0.f, w4 = 0.f;
        if (live) {
            const int kh = t / K, kw = t % K;
            const float h = (float)(oy - pad + kh) + offset[(long)(2 * t) * HW + pix];
            const float w = (float)(ox - pad + kw) + offset[(long)(2 * t + 1) * HW + pix];
            const float m = mask[(long)t * HW + pix];
            if (!(h <= -1.f || (float)H <= h || w <= -1.f || (float)W <= w)) {
                const int hl = (int)floorf(h), wl = (int)floorf(w), hh = hl + 1, wh = wl + 1;
                const float lh = h - hl, lw = w - wl, uh = 1.f - lh, uw = 1.f - lw;
                if (hl >= 0 && wl >= 0)         { i1 = (long)hl * W + wl; w1 = m * uh * uw; }
                if (hl >= 0 && wh <= W - 1)     { i2 = (long)hl * W + wh; w2 = m * uh * lw; }
                if (hh <= H - 1 && wl >= 0)     { i3 = (long)hh * W + wl; w3 = m * lh * uw; }
                if (hh <= H - 1 && wh <= W - 1) { i4 = (long)hh * W + wh; w4 = m * lh * lw; }
            }
        }
        for (int ic0 = 0; ic0 < Cin; ic0 += DC_ICT) {
            const int nic = min(DC_ICT, Cin - ic0);
            __syncthreads();
            for (int i = threadIdx.x; i < noc * nic; i += blockDim.x) {
                const int o = i / nic, ic = i % nic;
                ws[o * DC_ICT + ic] = weight[((long)(oc0 + o) * Cin + ic0 + ic) * K2 + t];
            }
            __syncthreads();
            if (!live) continue;
            for (int ic = 0; ic < nic; ++ic) {
                const float* xim = x + (long)(ic0 + ic) * HW;
                const float v = w1 * xim[i1] + w2 * xim[i2] + w3 * xim[i3] + w4 * xim[i4];
#pragma unroll
                for (int o = 0; o < DC_OCT; ++o) acc[o] += ws[o * DC_ICT + ic] * v;
            }
        }
    }
    if (!live) return;
    for (int o = 0; o < noc; ++o) out[(long)(oc0 + o) * HW + pix] = acc[o];
}

void deform_conv2d_run(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int gpu) {
    cudaSetDevice(gpu < 0 ? 0 : gpu);   // deform always runs on a real GPU even if the model is on CPU
    long HW = (long)H * W;
    size_t sx = (size_t)Cin * HW * sizeof(float);
    size_t soff = (size_t)2 * K * K * HW * sizeof(float);
    size_t smask = (size_t)K * K * HW * sizeof(float);
    size_t sw = (size_t)Cout * Cin * K * K * sizeof(float);
    size_t sout = (size_t)Cout * HW * sizeof(float);
    float *dx, *doff, *dmask, *dw, *dbias = nullptr, *dout;
    cudaMalloc(&dx, sx); cudaMalloc(&doff, soff); cudaMalloc(&dmask, smask);
    cudaMalloc(&dw, sw); cudaMalloc(&dout, sout);
    cudaMemcpy(dx, x, sx, cudaMemcpyHostToDevice);
    cudaMemcpy(doff, offset, soff, cudaMemcpyHostToDevice);
    cudaMemcpy(dmask, mask, smask, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, weight, sw, cudaMemcpyHostToDevice);
    if (bias) { cudaMalloc(&dbias, (size_t)Cout * sizeof(float)); cudaMemcpy(dbias, bias, (size_t)Cout * sizeof(float), cudaMemcpyHostToDevice); }
    dim3 grid((unsigned)((HW + DC_THREADS - 1) / DC_THREADS), (unsigned)((Cout + DC_OCT - 1) / DC_OCT));
    deform_conv_kernel<<<grid, DC_THREADS>>>(dx, Cin, H, W, doff, dmask, dw, dbias, Cout, K, K/2, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fprintf(stderr, "deform_conv kernel: %s\n", cudaGetErrorString(e));
    cudaMemcpy(out, dout, sout, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(doff); cudaFree(dmask); cudaFree(dw); cudaFree(dout);
    if (dbias) cudaFree(dbias);
}

} // namespace trellis
