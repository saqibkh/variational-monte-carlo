#pragma once

#include <xpu/config.hpp>
#include "macros.hpp"

#include <cstddef>

namespace execution {

[[nodiscard]] XPU_CUDA_CALLABLE
inline std::size_t thread() noexcept {
#if defined(__CUDA_ARCH__)
  return scast<std::size_t>(threadIdx.x);
#else
  return 0uz;
#endif
}

[[nodiscard]] XPU_CUDA_CALLABLE
inline std::size_t stride() noexcept {
#if defined(__CUDA_ARCH__)
  return scast<std::size_t>(blockDim.x);
#else
  return 1uz;
#endif
}

XPU_CUDA_CALLABLE
inline void sync() noexcept {
#if defined(__CUDA_ARCH__)
  __syncthreads();
#endif
}

} // namespace execution
