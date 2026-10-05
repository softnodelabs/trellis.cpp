// Modulated deformable conv2d (torchvision deform_conv2d v2) — custom CUDA op for BiRefNet's
// ASPPDeformable. Self-contained: host arrays in/out (NCHW, C-order), device work internally.
#pragma once

namespace trellis {

// x:[Cin,H,W]  offset:[2*K*K,H,W]  mask:[K*K,H,W] (already 2*sigmoid'd)  weight:[Cout,Cin,K,K]
// bias:[Cout] or nullptr.  stride=1, dilation=1, padding=K/2 -> out:[Cout,H,W].
// gpu = device index. Backend: CUDA kernel, else a Vulkan compute shader, else CPU.
void deform_conv2d_run(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int gpu);

// Whether deform_conv2d_run has a GPU kernel in this build (CUDA, HIP or Vulkan). Without one,
// BiRefNet on a GPU backend runs the convolution as ggml ops in its own graph instead
// (birefnet.cpp), since the host loop is minutes at the decoder's larger scales.
bool deform_conv2d_has_gpu_kernel();

// Portable CPU implementation. Used directly on pure-CPU builds and as the Vulkan
// path's fallback when no compute device is usable.
void deform_conv2d_cpu(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int gpu);

} // namespace trellis
