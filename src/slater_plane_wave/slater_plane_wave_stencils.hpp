#pragma once

#include <cstdlib>
#include <xpu/launch.hpp>
#include <xpu/math.hpp>
#include "slater_plane_wave.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

namespace stencil {
namespace slater {

template <typename PositionType, typename KVectorType>
CUDA_CALLABLE
inline void update_trig_cache(
  std::size_t i, std::size_t offset, std::size_t particle,
  xpu::soa_view<PositionType, idx(Axis::NUM)> particle_pos,
  xpu::soa_view<KVectorType, idx(Axis::NUM)> k_vector,
  fp_t* RESTRICT sin_cache, fp_t* RESTRICT cos_cache
) {
  const auto dot{
    k_vector[idx(Axis::X)][i] * particle_pos[idx(Axis::X)][particle] +
    k_vector[idx(Axis::Y)][i] * particle_pos[idx(Axis::Y)][particle] +
    k_vector[idx(Axis::Z)][i] * particle_pos[idx(Axis::Z)][particle]
  };

  const auto idx{offset + i};
  xpu::sincos(dot, &sin_cache[idx], &cos_cache[idx]);
}

CUDA_CALLABLE
inline void build_row(
  std::size_t i, std::size_t particle,
  std::size_t trig_row_stride,
  const fp_t* RESTRICT sin_cache,
  const fp_t* RESTRICT cos_cache,
  const std::size_t* RESTRICT orbital_k_index,
  const std::uint8_t* RESTRICT orbital_type,
  fp_t* RESTRICT new_row
) {
  const auto trig_idx{
    particle * trig_row_stride + orbital_k_index[i]
  };
  const auto type{scast<fp_t>(orbital_type[i])};
  const auto sin_term{sin_cache[trig_idx]};
  const auto cos_term{cos_cache[trig_idx]};

  new_row[i] = cos_term + type * (sin_term - cos_term);
}

CUDA_CALLABLE
inline void initialize_matrix(
  std::size_t orbital,
  std::size_t particle,
  SlaterPlaneWave::View slater
) {
  const auto offset{particle * slater.matrix_row_stride};

  build_row(
    orbital,
    particle,
    slater.trig_row_stride,
    slater.sin_cache,
    slater.cos_cache,
    slater.orbital_k_index,
    slater.orbital_type,
    slater.determinant + offset
  );

  slater.lower_upper[offset + orbital] = slater.determinant[offset + orbital];
}

[[nodiscard]] CUDA_CALLABLE
inline fp_t compute_log_abs_det(
  std::size_t orbital,
  std::size_t matrix_row_stride,
  const fp_t* RESTRICT lower_upper
) {
  const auto diagonal{
    lower_upper[orbital * matrix_row_stride + orbital]
  };

  return xpu::log(xpu::abs(diagonal));
}

[[nodiscard]] CUDA_CALLABLE
inline fp_t determinant_ratio_contribution(
  std::size_t orbital,
  const fp_t* RESTRICT new_row,
  const fp_t* RESTRICT inverse_row
) {
  return new_row[orbital] * inverse_row[orbital];
}

CUDA_CALLABLE
inline void determinant_ratio(
  std::size_t orbital,
  std::size_t particle,
  std::size_t matrix_row_stride,
  const fp_t* RESTRICT new_row,
  const fp_t* RESTRICT inv_det,
  fp_t* RESTRICT ratio
) {
  const auto contribution{
    determinant_ratio_contribution(
      orbital,
      new_row,
      inv_det + particle * matrix_row_stride
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(ratio, contribution);
#else
  *ratio += contribution;
#endif
}

CUDA_CALLABLE
inline void add_derivatives(
  std::size_t particle,
  std::size_t orbital,
  SlaterPlaneWave::View slater,
  Particles::View particles
) {
  const auto k_idx{slater.orbital_k_index[orbital]};
  const auto k_x{slater.k_vector[idx(Axis::X)][k_idx]};
  const auto k_y{slater.k_vector[idx(Axis::Y)][k_idx]};
  const auto k_z{slater.k_vector[idx(Axis::Z)][k_idx]};
  const auto k_mag{
    k_x * k_x +
    k_y * k_y +
    k_z * k_z
  };

  const auto type{scast<fp_t>(slater.orbital_type[orbital])};
  const auto trig_idx{particle * slater.trig_row_stride + k_idx};
  const auto sin_term{slater.sin_cache[trig_idx]};
  const auto cos_term{slater.cos_cache[trig_idx]};
  const auto weight{
    slater.inv_determinant[particle * slater.matrix_row_stride + orbital]
  };

  const auto gradient_factor{
    weight * (
      -sin_term + type * (sin_term + cos_term)
    )
  };
  const auto laplacian_factor{
    weight * (
      -cos_term + type * (cos_term - sin_term)
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(&particles.derivatives[idx(Derivatives::GRAD_X)][particle], gradient_factor * k_x);
  atomicAdd(&particles.derivatives[idx(Derivatives::GRAD_Y)][particle], gradient_factor * k_y);
  atomicAdd(&particles.derivatives[idx(Derivatives::GRAD_Z)][particle], gradient_factor * k_z);
  atomicAdd(&particles.derivatives[idx(Derivatives::LAP)][particle], laplacian_factor * k_mag);
#else
  particles.derivatives[idx(Derivatives::GRAD_X)][particle] += gradient_factor * k_x;
  particles.derivatives[idx(Derivatives::GRAD_Y)][particle] += gradient_factor * k_y;
  particles.derivatives[idx(Derivatives::GRAD_Z)][particle] += gradient_factor * k_z;
  particles.derivatives[idx(Derivatives::LAP)][particle] += laplacian_factor * k_mag;
#endif
}

CUDA_CALLABLE
inline void accumulate_derivatives(
  std::size_t i,
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives
) {
  const auto gradient_x{derivatives[idx(Derivatives::GRAD_X)][i]};
  const auto gradient_y{derivatives[idx(Derivatives::GRAD_Y)][i]};
  const auto gradient_z{derivatives[idx(Derivatives::GRAD_Z)][i]};
  const auto gradient_magnitude{
    gradient_x * gradient_x +
    gradient_y * gradient_y +
    gradient_z * gradient_z
  };

  derivatives[idx(Derivatives::LAP)][i] -= gradient_magnitude;
}

CUDA_CALLABLE
inline void k_update_inverse(
  std::size_t i, std::size_t j,
  std::size_t particle,
  std::size_t row_stride, fp_t inv_ratio,
  const fp_t* RESTRICT inv_d_col,
  const fp_t* RESTRICT solution_arr,
  fp_t* RESTRICT inv_det
) {
  const auto idx{j * row_stride + i};
  const auto factor{inv_d_col[i] * inv_ratio};

  if (j == particle) {
    inv_det[idx] = factor;
  } else {
    inv_det[idx] -= factor * solution_arr[j];
  }
}

CUDA_CALLABLE
inline void k_compute_sk(
  std::size_t i, std::size_t j,
  std::size_t row_stride,
  const fp_t* RESTRICT new_row,
  const fp_t* RESTRICT inv_det,
  fp_t* RESTRICT solution
) {
  const auto product{new_row[i] * inv_det[j * row_stride + i]};

#if defined(__CUDA_ARCH__)
  atomicAdd(solution, product);
#else
  *solution += product;
#endif
}

CUDA_CALLABLE
inline void update_trig_cache(
  std::size_t k_index,
  std::size_t particle,
  SlaterPlaneWave::View slater,
  Particles::View particles
) {
  update_trig_cache(
    k_index,
    particle * slater.trig_row_stride,
    particle,
    particles.pos,
    slater.k_vector,
    slater.sin_cache,
    slater.cos_cache
  );
}

CUDA_CALLABLE
inline void build_row(
  std::size_t orbital,
  std::size_t particle,
  SlaterPlaneWave::View slater
) {
  build_row(
    orbital,
    particle,
    slater.trig_row_stride,
    slater.sin_cache,
    slater.cos_cache,
    slater.orbital_k_index,
    slater.orbital_type,
    slater.new_row
  );
}

CUDA_CALLABLE
inline void determinant_ratio(
  std::size_t orbital,
  std::size_t particle,
  SlaterPlaneWave::View slater,
  fp_t* ratio
) {
  determinant_ratio(
    orbital,
    particle,
    slater.matrix_row_stride,
    slater.new_row,
    slater.inv_determinant,
    ratio
  );
}

CUDA_CALLABLE
inline void k_update_inverse(
  std::size_t column,
  std::size_t row,
  std::size_t particle,
  fp_t inverse_ratio,
  SlaterPlaneWave::View slater
) {
  k_update_inverse(
    column,
    row,
    particle,
    slater.matrix_row_stride,
    inverse_ratio,
    slater.inv_d_col,
    slater.solution,
    slater.inv_determinant
  );
}

} // namespace stencil::slater
} // namespace stencil
