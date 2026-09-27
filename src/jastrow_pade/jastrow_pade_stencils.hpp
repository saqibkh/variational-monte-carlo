#pragma once

#include <xpu/xpu.hpp>
#include "jastrow_pade.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

namespace stencil {
namespace jpade {

namespace {

XPU_CUDA_CALLABLE
inline fp_t evaluate_pair_value(
  const vec3_t& particle_pos,
  std::size_t other,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  vec3_view_t pos
) noexcept {
  vec3_t displacement{};

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

XPU_CUDA_CALLABLE
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

  const auto first_derivative{a * inverse_denominator_squared};
  const auto second_derivative{
    neg_two_ab * inverse_denominator_squared * inverse_denominator
  };

  *gradient_factor = first_derivative * inverse_distance;
  *laplacian_factor = second_derivative + 2.0_fp * first_derivative * inverse_distance;
}

XPU_CUDA_CALLABLE
inline void evaluate_pair_derivatives(
  const vec3_t& particle_pos,
  std::size_t other,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b, fp_t neg_two_ab,
  vec3_view_t pos,
  vec3_t* gradient,
  fp_t* laplacian
) noexcept {
  vec3_t displacement{};

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

XPU_CUDA_CALLABLE
inline void value(
  std::size_t other,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  const vec3_t& particle_pos,
  vec3_view_t pos,
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

XPU_CUDA_CALLABLE
inline void delta_value(
  std::size_t other,
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  const vec3_t& old_pos,
  const vec3_t& new_pos,
  vec3_view_t pos,
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

XPU_CUDA_CALLABLE
inline void compute_derivatives(
  std::size_t i,
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b, fp_t neg_two_ab,
  const vec3_t& old_pos,
  const vec3_t& new_pos,
  vec3_view_t pos,
  derivatives_view derivatives,
  fp_t* moved_gradient_x, fp_t* moved_gradient_y, fp_t* moved_gradient_z,fp_t* moved_laplacian
) noexcept {
  if (i == moved) { return; }

  vec3_t gradient_old{};
  vec3_t gradient_new{};

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
  atomicAdd(moved_gradient_x, gradient_new[idx(Axis::X)] - gradient_old[idx(Axis::X)]);
  atomicAdd(moved_gradient_y, gradient_new[idx(Axis::Y)] - gradient_old[idx(Axis::Y)]);
  atomicAdd(moved_gradient_z, gradient_new[idx(Axis::Z)] - gradient_old[idx(Axis::Z)]);
  atomicAdd(moved_laplacian, laplacian_new - laplacian_old);
#else
  *moved_gradient_x += gradient_new[idx(Axis::X)] - gradient_old[idx(Axis::X)];
  *moved_gradient_y += gradient_new[idx(Axis::Y)] - gradient_old[idx(Axis::Y)];
  *moved_gradient_z += gradient_new[idx(Axis::Z)] - gradient_old[idx(Axis::Z)];
  *moved_laplacian += laplacian_new - laplacian_old;
#endif
}

XPU_CUDA_CALLABLE
inline void add_derivatives(
  std::size_t particle,
  std::size_t other,
  JastrowPade::View jastrow,
  Particles::View particles,
  derivatives_view derivatives
) noexcept {
  const vec3_t particle_pos{
    particles.pos[idx(Axis::X)][particle],
    particles.pos[idx(Axis::Y)][particle],
    particles.pos[idx(Axis::Z)][particle]
  };
  vec3_t pair_gradient{};
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

XPU_CUDA_CALLABLE
inline void delta_value(
  std::size_t other,
  std::size_t moved,
  JastrowPade::View jastrow,
  Particles::View particles,
  const vec3_t& old_pos,
  const vec3_t& new_pos,
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
