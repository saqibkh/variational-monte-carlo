#pragma once

#include "macros.hpp"

#include <xpu/xpu.hpp>
#include <cstddef>

enum class Axis { X, Y, Z, NUM };
enum class Derivatives { GRAD_X, GRAD_Y, GRAD_Z, LAP, NUM };

template <typename T> [[nodiscard]] XPU_CUDA_CALLABLE
inline constexpr std::size_t idx(T idx) {
  return scast<std::size_t>(idx);
}

using vec3_t = xpu::array<fp_t, idx(Axis::NUM)>;
using vec3_view_t = xpu::soa_view<fp_t, idx(Axis::NUM)>;
using const_vec3_view_t = xpu::soa_view<const fp_t, idx(Axis::NUM)>;

using derivatives_view = xpu::soa_view<fp_t, idx(Derivatives::NUM)>;
using const_derivatives_view = xpu::soa_view<const fp_t, idx(Derivatives::NUM)>;
using derivatives_batch_view = xpu::soa_batch_view<fp_t, idx(Derivatives::NUM)>;

template <typename A, typename B> [[nodiscard]] XPU_CUDA_CALLABLE
inline constexpr std::size_t enum_index(A a, B b) {
  return idx(a) + idx(b);
}
