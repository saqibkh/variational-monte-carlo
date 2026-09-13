#pragma once

#include <xpu/xpu.hpp>
#include "jastrow_pade.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

#include "jastrow_pade_stencils.hpp"

namespace kernel {
namespace jpade {

namespace {

#if defined(XPU_CUDA)
__global__
void cudaValue(
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  vec3_view_t pos,
  fp_t* jastrow_value
) {
  const auto [i, j]{xpu::global_index<2>()};
  if (i >= pos.count() || j >= pos.count()) { return; }
  if (i >= j) { return; }

  vec3_t particle_pos{
    pos[idx(Axis::X)][i],
    pos[idx(Axis::Y)][i],
    pos[idx(Axis::Z)][i]
  };

  stencil::jpade::value(
    j,
    L, half_L, a, b,
    particle_pos, pos,
    jastrow_value
  );
}

__global__
void cudaDeltaValue(
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  vec3_t old_pos,
  vec3_view_t pos,
  fp_t* delta
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= pos.count()) { return; }

  vec3_t new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  stencil::jpade::delta_value(
    i, moved,
    L, half_L, a, b,
    old_pos, new_pos, pos,
    delta
  );
}

__global__
void cudaComputeDerivatives(
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b, fp_t neg_two_ab,
  vec3_t old_pos,
  vec3_view_t pos,
  derivatives_view derivatives
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= pos.count()) { return; }

  vec3_t new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  stencil::jpade::compute_derivatives(
    i, moved,
    L, half_L, a, b, neg_two_ab,
    old_pos, new_pos, pos, derivatives,
    &derivatives[idx(Derivatives::GRAD_X)][moved],
    &derivatives[idx(Derivatives::GRAD_Y)][moved],
    &derivatives[idx(Derivatives::GRAD_Z)][moved],
    &derivatives[idx(Derivatives::LAP)][moved]
  );
}

__global__
void cudaAddDerivatives(
  JastrowPade::View jastrow,
  Particles::View particles,
  derivatives_view derivatives
) {
  const auto [i, j]{xpu::global_index<2>()};
  if (i >= particles.count || j >= particles.count) { return; }
  if (i == j) { return; }

  stencil::jpade::add_derivatives(i, j, jastrow, particles, derivatives);
}
#endif

} // namespace

inline fp_t value(
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  vec3_view_t pos
) {
#if defined(XPU_CUDA)
  xpu::buffer<fp_t> jastrow_value{1};

  dim3 valueThreads{16, 16};
  dim3 valueBlocks{
    xpu::block_per_dim(pos.count(), valueThreads.x),
    xpu::block_per_dim(pos.count(), valueThreads.y)
  };

  cudaValue<<<
    valueBlocks, valueThreads
  >>>(
    L, half_L, a, b,
    pos, jastrow_value.data()
  );
  xpu::cu_check(cudaGetLastError());

  fp_t value_host{};
  xpu::cu_check(cudaMemcpy(
    &value_host, jastrow_value.data(),
    sizeof(fp_t), cudaMemcpyDeviceToHost
  ));
  return value_host;
#else
  auto jastrow_value{0.0_fp};

  for (auto i = 0uz; i < pos.count(); ++i) {
    vec3_t particle_pos{
      pos[idx(Axis::X)][i],
      pos[idx(Axis::Y)][i],
      pos[idx(Axis::Z)][i]
    };

    auto local_value{0.0_fp};

    #pragma omp simd reduction(+ : local_value)
    for (auto j = i + 1uz; j < pos.count(); ++j) {
      stencil::jpade::value(
        j,
        L, half_L, a, b,
        particle_pos, pos,
        &local_value
      );
    }

    jastrow_value += local_value;
  }

  return jastrow_value;
#endif
}

inline fp_t delta_value(
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b,
  vec3_t old_pos,
  vec3_view_t pos
) {
#if defined(XPU_CUDA)
  xpu::buffer<fp_t> delta{1};

  dim3 deltaValueThreads{256};
  dim3 deltaValueBlocks{
    xpu::block_per_dim(pos.count(), deltaValueThreads.x)
  };

  cudaDeltaValue<<<
    deltaValueBlocks, deltaValueThreads
  >>>(
    moved,
    L, half_L, a, b,
    old_pos, pos, delta.data()
  );
  xpu::cu_check(cudaGetLastError());

  fp_t delta_host{};
  xpu::cu_check(cudaMemcpy(
    &delta_host, delta.data(),
    sizeof(fp_t), cudaMemcpyDeviceToHost
  ));
  return delta_host;
#else
  vec3_t new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  auto delta{0.0_fp};

  #pragma omp simd reduction(+ : delta)
  for (auto i = 0uz; i < pos.count(); ++i) {
    stencil::jpade::delta_value(
      i, moved,
      L, half_L, a, b,
      old_pos, new_pos, pos,
      &delta
    );
  }

  return delta;
#endif
}

inline void compute_derivatives(
  std::size_t moved,
  fp_t L, fp_t half_L,
  fp_t a, fp_t b, fp_t neg_two_ab,
  vec3_t old_pos,
  vec3_view_t pos,
  derivatives_view derivatives
) {
#if defined(XPU_CUDA)
  dim3 computeDerivativesThreads{256};
  dim3 computeDerivativesBlocks{
    xpu::block_per_dim(pos.count(), computeDerivativesThreads.x)
  };

  cudaComputeDerivatives<<<
    computeDerivativesBlocks, computeDerivativesThreads
  >>>(
    moved, 
    L, half_L, a, b, neg_two_ab,
    old_pos, pos, derivatives
  );
  xpu::cu_check(cudaGetLastError());
#else
  vec3_t new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  auto& moved_gradient_x{derivatives[idx(Derivatives::GRAD_X)][moved]};
  auto& moved_gradient_y{derivatives[idx(Derivatives::GRAD_Y)][moved]};
  auto& moved_gradient_z{derivatives[idx(Derivatives::GRAD_Z)][moved]};
  auto& moved_laplacian{derivatives[idx(Derivatives::LAP)][moved]};

  #pragma omp simd reduction(+ : moved_gradient_x, moved_gradient_y, moved_gradient_z, moved_laplacian)
  for (auto i = 0uz; i < pos.count(); ++i) {
    stencil::jpade::compute_derivatives(
      i, moved,
      L, half_L, a, b, neg_two_ab,
      old_pos, new_pos, pos, derivatives,
      &moved_gradient_x, &moved_gradient_y, &moved_gradient_z, &moved_laplacian
    );
  }
#endif
}

inline void add_derivatives(
  JastrowPade::View jastrow,
  Particles::View particles,
  derivatives_view derivatives
) {
#if defined(XPU_CUDA)
  dim3 addDerivativesThreads{16, 16};
  dim3 addDerivativesBlocks{
    xpu::block_per_dim(particles.count, addDerivativesThreads.x),
    xpu::block_per_dim(particles.count, addDerivativesThreads.y)
  };

  cudaAddDerivatives<<<
    addDerivativesBlocks, addDerivativesThreads
  >>>(
    jastrow,
    particles,
    derivatives
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto i = 0uz; i < particles.count; ++i) {
    for (auto j = 0uz; j < particles.count; ++j) {
      stencil::jpade::add_derivatives(
        i, j, jastrow, particles, derivatives
      );
    }
  }
#endif
}

inline fp_t value(
  JastrowPade::View jastrow,
  Particles::View particles
) noexcept {
  return value(
    jastrow.box_length,
    0.5_fp * jastrow.box_length,
    jastrow.a,
    jastrow.b,
    particles.pos
  );
}

inline fp_t delta_value(
  JastrowPade::View jastrow,
  std::size_t moved,
  vec3_t old_pos,
  Particles::View particles
) noexcept {
  return delta_value(
    moved,
    jastrow.box_length,
    0.5_fp * jastrow.box_length,
    jastrow.a,
    jastrow.b,
    old_pos,
    particles.pos
  );
}

inline void compute_derivatives(
  JastrowPade::View jastrow,
  std::size_t moved,
  vec3_t old_pos,
  Particles::View particles,
  derivatives_view derivatives
) noexcept {
  compute_derivatives(
    moved,
    jastrow.box_length,
    0.5_fp * jastrow.box_length,
    jastrow.a,
    jastrow.b,
    -2.0_fp * jastrow.a * jastrow.b,
    old_pos,
    particles.pos,
    derivatives
  );
}

} // namespace kernel::jpade
} // namespace kernel
