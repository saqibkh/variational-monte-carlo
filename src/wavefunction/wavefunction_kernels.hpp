#pragma once

#include <xpu/xpu.hpp>
#include "wavefunction_stencils.hpp"
#include "wavefunction.hpp"

namespace kernel {
namespace wavefunction {

namespace {

#if defined(XPU_CUDA)
__global__
void cudaDerivativeSums(
  derivatives_view log_derivatives,
  derivatives_view jastrow_derivatives
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= log_derivatives.stride()) { return; }

  stencil::wavefunction::derivative_sum(i,log_derivatives, jastrow_derivatives);
}
#endif

} // namespace

inline void derivative_sums(
  derivatives_view log_derivatives,
  derivatives_view jastrow_derivatives
) {
#if defined(XPU_CUDA)
  dim3 derivativeSumsThreads{256};
  dim3 derivativeSumsBlocks{
    xpu::block_per_dim(log_derivatives.stride(), derivativeSumsThreads.x)
  };
  
  cudaDerivativeSums<<<
    derivativeSumsBlocks, derivativeSumsThreads
  >>>(
    log_derivatives,
    jastrow_derivatives
  );

  xpu::cu_check(cudaGetLastError());
  xpu::cu_check(cudaDeviceSynchronize());
#else
  const auto N{log_derivatives.stride()};
  #pragma omp simd
  for (auto i = 0uz; i < N; ++i) {
    stencil::wavefunction::derivative_sum(i, log_derivatives, jastrow_derivatives);
  }
#endif
}

inline void derivative_sums(
  WaveFunction::View wave_function,
  Particles::View particles
) {
  derivative_sums(
    particles.derivatives,
    wave_function.jastrow_derivatives
  );
}

} // namespace kernel::wavefunction
} // namespace kernel
