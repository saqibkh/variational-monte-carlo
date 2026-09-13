#pragma once

#include <xpu/xpu.hpp>
#include "jastrow_pade.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

namespace stencil {
namespace jpade {

namespace {

CUDA_CALLABLE
inline fp_t evaluate_pair_value(
  const xpu::array<fp_t, idx(Axis::NUM)>& particle_pos,
  std::size_t other,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos
) noexcept {
  xpu::array<fp_t, idx(Axis::NUM)> displacement{};

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    auto delta{particle_pos[axis] - pos[axis][other]};

    delta += L * (delta <= -half_L) - L * (delta > half_L);

    displacement[axis] = delta;
  }

  const auto distance{
    xpu::sqrt(
      displacement[idx(Axis::X)] * displacement[idx(Axis::X)] +
      displacement[idx(Axis::Y)] * displacement[idx(Axis::Y)] +
      displacement[idx(Axis::Z)] * displacement[idx(Axis::Z)]
    )
  };

  return a * distance / (1.0_fp + b * distance);
}

CUDA_CALLABLE
inline void evaluate_derivative_factors(
  const fp_t distance,
  const fp_t a,
  const fp_t b,
  const fp_t neg_two_ab,
  fp_t* gradient_factor,
  fp_t* laplacian_factor
) noexcept {
  if (distance < epsilon) {
    *gradient_factor = 0.0_fp;
    *laplacian_factor = 0.0_fp;
    return;
  }

  const auto inverse_distance{1.0_fp / distance};
  const auto inverse_denominator{1.0_fp / (1.0_fp + b * distance)};
  const auto inverse_denominator_squared{
    inverse_denominator * inverse_denominator
  };

  const auto first_deriv{a * inverse_denominator_squared};
  const auto second_deriv{
    neg_two_ab * inverse_denominator_squared * inverse_denominator
  };

  *gradient_factor = first_deriv * inverse_distance;
  *laplacian_factor = second_deriv + 2.0_fp * first_deriv * inverse_distance;
}

CUDA_CALLABLE
inline void evaluate_pair_derivatives(
  const xpu::array<fp_t, idx(Axis::NUM)>& particle_pos,
  std::size_t other,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b, fp_t neg_two_ab,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  xpu::array<fp_t, idx(Axis::NUM)>* gradient,
  fp_t* laplacian
) noexcept {
  xpu::array<fp_t, idx(Axis::NUM)> displacement{};

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    auto delta{particle_pos[axis] - pos[axis][other]};

    delta += L * (delta <= -half_L) - L * (delta > half_L);

    displacement[axis] = delta;
  }

  const auto distance{
    xpu::sqrt(
      displacement[idx(Axis::X)] * displacement[idx(Axis::X)] +
      displacement[idx(Axis::Y)] * displacement[idx(Axis::Y)] +
      displacement[idx(Axis::Z)] * displacement[idx(Axis::Z)]
    )
  };

  auto gradient_factor{0.0_fp};

  evaluate_derivative_factors(
    distance, a, b, neg_two_ab,
    &gradient_factor, laplacian
  );

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    (*gradient)[axis] = gradient_factor * displacement[axis];
  }
}

} // namespace

CUDA_CALLABLE
inline void value(
  std::size_t other,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  const xpu::array<fp_t, idx(Axis::NUM)>& particle_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* jastrow_value
) noexcept {
  const auto pair_value{
    evaluate_pair_value(
      particle_pos, other,
      L, half_L, a, b,
      pos
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(jastrow_value, pair_value);
#else
  *jastrow_value += pair_value;
#endif
}

CUDA_CALLABLE
inline void delta_value(
  std::size_t other,
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* delta
) noexcept {
  if (other == moved) { return; }

  const auto value_old{
    evaluate_pair_value(
      old_pos, other,
      L, half_L, a, b,
      pos
    )
  };
  const auto value_new{
    evaluate_pair_value(
      new_pos, other,
      L, half_L, a, b,
      pos
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(delta, value_new - value_old);
#else
  *delta += value_new - value_old;
#endif
}

CUDA_CALLABLE
inline void compute_derivatives(
  std::size_t i,
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b, fp_t neg_two_ab,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives,
  fp_t* moved_gx, fp_t* moved_gy, fp_t* moved_gz,fp_t* moved_lap
) noexcept {
  if (i == moved) { return; }

  xpu::array<fp_t, idx(Axis::NUM)> gradient_old{};
  xpu::array<fp_t, idx(Axis::NUM)> gradient_new{};

  auto laplacian_old{0.0_fp};
  auto laplacian_new{0.0_fp};

  evaluate_pair_derivatives(
    old_pos, i,
    L, half_L, a, b, neg_two_ab,
    pos, &gradient_old, &laplacian_old
  );
  evaluate_pair_derivatives(
    new_pos, i,
    L, half_L, a, b, neg_two_ab,
    pos, &gradient_new, &laplacian_new
  );

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    derivatives[axis][i] += gradient_old[axis] - gradient_new[axis];
  }
  derivatives[idx(Derivatives::LAP)][i] += laplacian_new - laplacian_old;

#if defined(__CUDA_ARCH__)
  atomicAdd(moved_gx, gradient_new[idx(Axis::X)] - gradient_old[idx(Axis::X)]);
  atomicAdd(moved_gy, gradient_new[idx(Axis::Y)] - gradient_old[idx(Axis::Y)]);
  atomicAdd(moved_gz, gradient_new[idx(Axis::Z)] - gradient_old[idx(Axis::Z)]);
  atomicAdd(moved_lap, laplacian_new - laplacian_old);
#else
  *moved_gx += gradient_new[idx(Axis::X)] - gradient_old[idx(Axis::X)];
  *moved_gy += gradient_new[idx(Axis::Y)] - gradient_old[idx(Axis::Y)];
  *moved_gz += gradient_new[idx(Axis::Z)] - gradient_old[idx(Axis::Z)];
  *moved_lap += laplacian_new - laplacian_old;
#endif
}

CUDA_CALLABLE
inline void add_derivatives(
  std::size_t particle,
  std::size_t other,
  JastrowPade::View jastrow,
  Particles::View particles,
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives
) noexcept {
  const xpu::array<fp_t, idx(Axis::NUM)> particle_pos{
    particles.pos[idx(Axis::X)][particle],
    particles.pos[idx(Axis::Y)][particle],
    particles.pos[idx(Axis::Z)][particle]
  };
  xpu::array<fp_t, idx(Axis::NUM)> pair_gradient{};
  auto pair_laplacian{0.0_fp};

  evaluate_pair_derivatives(
    particle_pos, other,
    jastrow.box_length,
    0.5_fp * jastrow.box_length,
    jastrow.a,
    jastrow.b,
    -2.0_fp * jastrow.a * jastrow.b,
    particles.pos,
    &pair_gradient,
    &pair_laplacian
  );

#if defined(__CUDA_ARCH__)
  atomicAdd(&derivatives[idx(Derivatives::GRAD_X)][particle], pair_gradient[idx(Axis::X)]);
  atomicAdd(&derivatives[idx(Derivatives::GRAD_Y)][particle], pair_gradient[idx(Axis::Y)]);
  atomicAdd(&derivatives[idx(Derivatives::GRAD_Z)][particle], pair_gradient[idx(Axis::Z)]);
  atomicAdd(&derivatives[idx(Derivatives::LAP)][particle], pair_laplacian);
#else
  derivatives[idx(Derivatives::GRAD_X)][particle] += pair_gradient[idx(Axis::X)];
  derivatives[idx(Derivatives::GRAD_Y)][particle] += pair_gradient[idx(Axis::Y)];
  derivatives[idx(Derivatives::GRAD_Z)][particle] += pair_gradient[idx(Axis::Z)];
  derivatives[idx(Derivatives::LAP)][particle] += pair_laplacian;
#endif
}

CUDA_CALLABLE
inline void delta_value(
  std::size_t other,
  std::size_t moved,
  JastrowPade::View jastrow,
  Particles::View particles,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  fp_t* delta
) noexcept {
  delta_value(
    other,
    moved,
    jastrow.box_length,
    0.5_fp * jastrow.box_length,
    jastrow.a,
    jastrow.b,
    old_pos,
    new_pos,
    particles.pos,
    delta
  );
}

} // namespace stencil::jpade
} // namespace stencil
