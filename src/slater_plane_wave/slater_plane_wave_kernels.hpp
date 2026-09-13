#pragma once

#include <xpu/launch.hpp>
#include <xpu/math.hpp>
#include "slater_plane_wave.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

#include "slater_plane_wave_stencils.hpp"

namespace kernel {
namespace slater {

namespace {

struct DeterminantRatioContribution {
  const fp_t* new_row{};
  const fp_t* inverse_row{};

  [[nodiscard]] CUDA_CALLABLE
  fp_t operator()(const xpu::array<std::size_t, 1uz>& index) const {
    return stencil::slater::determinant_ratio_contribution(
      index[0uz],
      new_row,
      inverse_row
    );
  }
};

struct LogAbsDetContribution {
  const fp_t* lower_upper{};
  std::size_t matrix_row_stride{};

  [[nodiscard]] CUDA_CALLABLE
  fp_t operator()(const xpu::array<std::size_t, 1uz>& index) const {
    return stencil::slater::compute_log_abs_det(
      index[0uz],
      matrix_row_stride,
      lower_upper
    );
  }
};

#if defined(XPU_CUDA)
constexpr auto initialization_columns{32uz};
constexpr auto initialization_rows{8uz};

__global__
void cudaInitializeTrigCache(
  SlaterPlaneWave::BatchView slaters,
  Particles::BatchView particles
) {
  const auto [k_index, particle, walker]{xpu::global_index<3>()};

  if (
    k_index >= slaters.num_unique_k ||
    particle >= slaters.num_orbitals ||
    walker >= slaters.walker_count()
  ) { return; }

  stencil::slater::update_trig_cache(
    k_index,
    particle,
    slaters.view(walker),
    particles.view(walker)
  );
}

__global__
void cudaInitializeMatrices(
  SlaterPlaneWave::BatchView slaters
) {
  const auto [orbital, particle, walker]{xpu::global_index<3>()};

  if (
    orbital >= slaters.num_orbitals ||
    particle >= slaters.num_orbitals ||
    walker >= slaters.walker_count()
  ) { return; }

  stencil::slater::initialize_matrix(
    orbital,
    particle,
    slaters.view(walker)
  );
}

__global__
void cudaUpdateTrigRow(
  std::size_t num_unique_k,
  std::size_t offset, std::size_t particle,
  vec3_view_t particle_pos,
  vec3_view_t k_vector,
  fp_t* RESTRICT sin_cache, fp_t* RESTRICT cos_cache
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= num_unique_k) { return; }

  stencil::slater::update_trig_cache(
    i, offset, particle,
    particle_pos, k_vector,
    sin_cache, cos_cache
  );
}

__global__
void cudaBuildTrigCache(
  std::size_t num_unique_k,
  std::size_t trig_row_stride,
  vec3_view_t particle_pos,
  vec3_view_t k_vector,
  fp_t* RESTRICT sin_cache,
  fp_t* RESTRICT cos_cache
) {
  const auto [i, j]{xpu::global_index<2>()};
  if (i >= num_unique_k || j >= particle_pos.count()) { return; }

  stencil::slater::update_trig_cache(
    i, j * trig_row_stride, j,
    particle_pos, k_vector,
    sin_cache, cos_cache
  );
}

__global__
void cudaBuildRow(
  std::size_t num_orbitals, std::size_t particle,
  std::size_t trig_row_stride,
  const fp_t* RESTRICT sin_cache,
  const fp_t* RESTRICT cos_cache,
  const std::size_t* RESTRICT orbital_k_index,
  const std::uint8_t* RESTRICT orbital_type,
  fp_t* RESTRICT new_row
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= num_orbitals) { return; }

  stencil::slater::build_row(
    i, particle, trig_row_stride,
    sin_cache, cos_cache,
    orbital_k_index, orbital_type,
    new_row
  );
}

__global__
void cudaBuildDeterminant(
  std::size_t num_orbitals,
  std::size_t trig_row_stride,
  std::size_t matrix_row_stride,
  const fp_t* RESTRICT sin_cache,
  const fp_t* RESTRICT cos_cache,
  const std::size_t* RESTRICT orbital_k_index,
  const std::uint8_t* RESTRICT orbital_type,
  fp_t* RESTRICT determinant
) {
  const auto [i, j]{xpu::global_index<2>()};
  if (i >= num_orbitals || j >= num_orbitals) { return; }

  stencil::slater::build_row(
    i, j, trig_row_stride,
    sin_cache, cos_cache,
    orbital_k_index, orbital_type,
    &determinant[j * matrix_row_stride]
  );
}

__global__
void cudaAddDerivatives(
  SlaterPlaneWave::View slater,
  Particles::View particles
) {
  const auto [i, j]{xpu::global_index<2>()};
  if (i >= slater.num_orbitals || j >= slater.num_orbitals) { return; }

  stencil::slater::add_derivatives(i, j, slater, particles);
}

__global__
void cudaAccumulateDerivatives(
  derivatives_view derivatives
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= derivatives.count()) { return; }

  stencil::slater::accumulate_derivatives(i, derivatives);
}

__global__
void kUpdateInverse(
  std::size_t num_orbitals, std::size_t particle,
  std::size_t row_stride, fp_t inv_ratio,
  const fp_t* RESTRICT inv_d_col,
  const fp_t* RESTRICT solution_arr,
  fp_t* RESTRICT inv_det
) {
  const auto [i,j]{xpu::global_index<2>()};
  if (i >= num_orbitals || j >= num_orbitals) { return; }

  stencil::slater::k_update_inverse(
    i, j,
    particle, row_stride, inv_ratio,
    inv_d_col, solution_arr, inv_det
  );
}

__global__
void kComputeSK(
  std::size_t num_orbitals, std::size_t particle,
  std::size_t row_stride,
  const fp_t* RESTRICT new_row,
  const fp_t* RESTRICT inv_det,
  fp_t* RESTRICT solution_arr
) {
  const auto [i, j]{xpu::global_index<2>()};
  if (i >= num_orbitals || j >= num_orbitals) { return; }
  if (j == particle) { return; }

  stencil::slater::k_compute_sk(
    i, j, row_stride,
    new_row, inv_det,
    &solution_arr[j]
  );
}
#endif

} // namespace

inline void initialize_matrices(
  SlaterPlaneWave::BatchView slaters,
  Particles::BatchView particles,
  [[maybe_unused]] std::size_t num_threads
) {
  const auto walker_count{slaters.walker_count()};
  if (walker_count == 0uz || slaters.num_orbitals == 0uz) { return; }

#if defined(XPU_CUDA)
  constexpr dim3 initializeTrigCacheThreads{
    scast<unsigned int>(initialization_columns),
    scast<unsigned int>(initialization_rows),
    1u
  };
  const dim3 initializeTrigCacheBlocks{
    xpu::block_per_dim(slaters.num_unique_k, initializeTrigCacheThreads.x),
    xpu::block_per_dim(slaters.num_orbitals, initializeTrigCacheThreads.y),
    scast<unsigned int>(walker_count)
  };

  cudaInitializeTrigCache<<<
    initializeTrigCacheBlocks, initializeTrigCacheThreads
  >>>(slaters, particles);
  xpu::cu_check(cudaGetLastError());

  constexpr dim3 initializeMatricesThreads{
    scast<unsigned int>(initialization_columns),
    scast<unsigned int>(initialization_rows),
    1u
  };
  const dim3 initializeMatricesBlocks{
    xpu::block_per_dim(slaters.num_orbitals, initializeMatricesThreads.x),
    xpu::block_per_dim(slaters.num_orbitals, initializeMatricesThreads.y),
    scast<unsigned int>(walker_count)
  };

  cudaInitializeMatrices<<<
    initializeMatricesBlocks, initializeMatricesThreads
  >>>(slaters);
  xpu::cu_check(cudaGetLastError());
#else
  #pragma omp parallel for num_threads(num_threads)
  for (auto walker = 0uz; walker < walker_count; ++walker) {
    const auto slater{slaters.view(walker)};
    const auto walker_particles{particles.view(walker)};

    for (auto particle = 0uz; particle < slater.num_orbitals; ++particle) {
      #pragma omp simd
      for (auto k_index = 0uz; k_index < slater.num_unique_k; ++k_index) {
        stencil::slater::update_trig_cache(
          k_index, particle, slater, walker_particles
        );
      }

      #pragma omp simd
      for (auto orbital = 0uz; orbital < slater.num_orbitals; ++orbital) {
        stencil::slater::initialize_matrix(
          orbital, particle, slater
        );
      }
    }
  }
#endif
}

inline void update_trig_cache(
  std::size_t num_unique_k,
  std::size_t offset, std::size_t particle,
  vec3_view_t particle_pos,
  vec3_view_t k_vector,
  fp_t* RESTRICT sin_cache, fp_t* RESTRICT cos_cache
) {
#if defined(XPU_CUDA)
  dim3 updateTrigRowThreads{256u};
  dim3 updateTrigRowBlocks{
    xpu::block_per_dim(num_unique_k, updateTrigRowThreads.x)
  };
  cudaUpdateTrigRow<<<
    updateTrigRowBlocks, updateTrigRowThreads
  >>>(
    num_unique_k,
    offset, particle,
    particle_pos, k_vector,
    sin_cache, cos_cache
  );
  xpu::cu_check(cudaGetLastError());
  xpu::cu_check(cudaDeviceSynchronize());
#else
  #pragma omp simd
  for (auto i = 0uz; i < num_unique_k; ++i) {
    stencil::slater::update_trig_cache(
      i, offset, particle,
      particle_pos, k_vector,
      sin_cache, cos_cache
    );
  }
#endif
}

inline void build_trig_cache(
  std::size_t num_unique_k,
  std::size_t trig_row_stride,
  vec3_view_t particle_pos,
  vec3_view_t k_vector,
  fp_t* RESTRICT sin_cache,
  fp_t* RESTRICT cos_cache
) {
#if defined(XPU_CUDA)
  dim3 buildTrigCacheThreads{16u, 16u};
  dim3 buildTrigCacheBlocks(
    xpu::block_per_dim(num_unique_k, buildTrigCacheThreads.x),
    xpu::block_per_dim(particle_pos.count(), buildTrigCacheThreads.y)
  );
  cudaBuildTrigCache<<<
    buildTrigCacheBlocks, buildTrigCacheThreads
  >>>(
    num_unique_k, trig_row_stride,
    particle_pos, k_vector,
    sin_cache, cos_cache
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto j = 0uz; j < particle_pos.count(); ++j) {
    const auto offset{j * trig_row_stride};

    #pragma omp simd
    for (auto i = 0uz; i < num_unique_k; ++i) {
      stencil::slater::update_trig_cache(
        i, offset, j,
        particle_pos, k_vector,
        sin_cache, cos_cache
      );
    }
  }
#endif
}

inline void build_row(
  std::size_t num_orbitals, std::size_t particle,
  std::size_t trig_row_stride,
  const fp_t* RESTRICT sin_cache,
  const fp_t* RESTRICT cos_cache,
  const std::size_t* RESTRICT orbital_k_index,
  const std::uint8_t* RESTRICT orbital_type,
  fp_t* RESTRICT new_row
) {
#if defined(XPU_CUDA)
  dim3 buildRowThreads{256u};
  dim3 buildRowBlocks{
    xpu::block_per_dim(num_orbitals, buildRowThreads.x)
  };
  cudaBuildRow<<<
    buildRowBlocks, buildRowThreads
  >>>(
    num_orbitals, particle, trig_row_stride,
    sin_cache, cos_cache,
    orbital_k_index, orbital_type,
    new_row
  );
  xpu::cu_check(cudaGetLastError());
#else
  #pragma omp simd
  for (auto i = 0uz; i < num_orbitals; ++i) {
    stencil::slater::build_row(
      i, particle, trig_row_stride,
      sin_cache, cos_cache,
      orbital_k_index, orbital_type,
      new_row
    );
  }
#endif
}

inline void build_determinant(
  std::size_t num_orbitals,
  std::size_t trig_row_stride,
  std::size_t matrix_row_stride,
  const fp_t* RESTRICT sin_cache,
  const fp_t* RESTRICT cos_cache,
  const std::size_t* RESTRICT orbital_k_index,
  const std::uint8_t* RESTRICT orbital_type,
  fp_t* RESTRICT determinant
) {
#if defined(XPU_CUDA)
  dim3 buildDeterminantThreads{16u, 16u};
  dim3 buildDeterminantBlocks(
    xpu::block_per_dim(num_orbitals, buildDeterminantThreads.x),
    xpu::block_per_dim(num_orbitals, buildDeterminantThreads.y)
  );
  cudaBuildDeterminant<<<
    buildDeterminantBlocks, buildDeterminantThreads
  >>>(
    num_orbitals,
    trig_row_stride, matrix_row_stride,
    sin_cache, cos_cache,
    orbital_k_index, orbital_type,
    determinant
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto j = 0uz; j < num_orbitals; ++j) {
    #pragma omp simd
    for (auto i = 0uz; i < num_orbitals; ++i) {
      stencil::slater::build_row(
        i, j, trig_row_stride,
        sin_cache, cos_cache,
        orbital_k_index, orbital_type,
        &determinant[j * matrix_row_stride]
      );
    }
  }
#endif
}

[[nodiscard]]
inline std::size_t log_abs_det_scratch_bytes(
  std::size_t num_orbitals,
  std::size_t matrix_row_stride
) {
  const xpu::range<1uz> range{
    {0uz},
    {num_orbitals},
    {1uz}
  };
  const LogAbsDetContribution contribution{
    nullptr,
    matrix_row_stride
  };

  return xpu::parallel_reduce_sum_bytes<fp_t>(range, contribution);
}

inline fp_t compute_log_abs_det(
  SlaterPlaneWave::View slater,
  void* scratch,
  std::size_t scratch_bytes
) {
  const xpu::range<1uz> range{
    {0uz},
    {slater.num_orbitals},
    {1uz}
  };
  const LogAbsDetContribution contribution{
    slater.lower_upper,
    slater.matrix_row_stride
  };

  xpu::parallel_reduce_sum(
    range,
    slater.reduction_scratch,
    contribution,
    scratch,
    scratch_bytes
  );

  auto log_abs_det{0.0_fp};
  xpu::copy_n(&log_abs_det, slater.reduction_scratch, 1uz);
  return log_abs_det;
}

[[nodiscard]]
inline std::size_t determinant_ratio_scratch_bytes(
  std::size_t num_orbitals
) {
  const xpu::range<1uz> range{
    {0uz},
    {num_orbitals},
    {1uz}
  };
  const DeterminantRatioContribution contribution{};

  return xpu::parallel_reduce_sum_bytes<fp_t>(range, contribution);
}

inline fp_t determinant_ratio(
  SlaterPlaneWave::View slater,
  std::size_t particle,
  const fp_t* new_row,
  void* scratch,
  std::size_t scratch_bytes
) {
  const xpu::range<1uz> range{
    {0uz},
    {slater.num_orbitals},
    {1uz}
  };
  const DeterminantRatioContribution contribution{
    new_row,
    slater.inv_determinant + particle * slater.matrix_row_stride
  };

  xpu::parallel_reduce_sum(
    range,
    slater.reduction_scratch,
    contribution,
    scratch,
    scratch_bytes
  );

  auto ratio{0.0_fp};
  xpu::copy_n(&ratio, slater.reduction_scratch, 1uz);
  return ratio;
}

inline void add_derivatives(
  SlaterPlaneWave::View slater,
  Particles::View particles
) {
#if defined(XPU_CUDA)
  dim3 addDerivativesThreads{16u, 16u};
  dim3 addDerivativesBlocks(
    xpu::block_per_dim(slater.num_orbitals, addDerivativesThreads.x),
    xpu::block_per_dim(slater.num_orbitals, addDerivativesThreads.y)
  );
  cudaAddDerivatives<<<
    addDerivativesBlocks, addDerivativesThreads
  >>>(
    slater,
    particles
  );
  xpu::cu_check(cudaGetLastError());

  dim3 accumulateDerivativesThreads{256u};
  dim3 accumulateDerivativesBlocks{
    xpu::block_per_dim(slater.num_orbitals, accumulateDerivativesThreads.x)
  };
  cudaAccumulateDerivatives<<<
    accumulateDerivativesBlocks, accumulateDerivativesThreads
  >>>(
    particles.derivatives
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto i = 0uz; i < slater.num_orbitals; ++i) {
    for (auto j = 0uz; j < slater.num_orbitals; ++j) {
      stencil::slater::add_derivatives(i, j, slater, particles);
    }

    stencil::slater::accumulate_derivatives(i, particles.derivatives);
  }
#endif
}

inline void k_update_inverse(
  std::size_t num_orbitals, std::size_t particle,
  std::size_t row_stride, fp_t inv_ratio,
  const fp_t* RESTRICT inv_d_col,
  const fp_t* RESTRICT solution_arr,
  fp_t* RESTRICT inv_det
) {
#if defined(XPU_CUDA)
  dim3 kUpdateInverseThreads{16u, 16u};
  dim3 kUpdateInverseBlocks(
    xpu::block_per_dim(num_orbitals, kUpdateInverseThreads.x),
    xpu::block_per_dim(num_orbitals, kUpdateInverseThreads.y)
  );
  kUpdateInverse<<<
    kUpdateInverseBlocks, kUpdateInverseThreads
  >>>(
    num_orbitals, particle, row_stride, inv_ratio,
    inv_d_col, solution_arr, inv_det
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto j = 0uz; j < num_orbitals; ++j) {
    #pragma omp simd
    for (auto i = 0uz; i < num_orbitals; ++i) {
      stencil::slater::k_update_inverse(
        i, j,
        particle, row_stride, inv_ratio,
        inv_d_col, solution_arr, inv_det
      );
    }
  }
#endif
}

inline void k_compute_sk(
  std::size_t num_orbitals, std::size_t particle,
  std::size_t row_stride,
  const fp_t* RESTRICT new_row,
  const fp_t* RESTRICT inv_det,
  fp_t* RESTRICT solution_arr
) {
#if defined(XPU_CUDA)
  dim3 kComputeSKThreads{16u, 16u};
  dim3 kComputeSKBlocks(
    xpu::block_per_dim(num_orbitals, kComputeSKThreads.x),
    xpu::block_per_dim(num_orbitals, kComputeSKThreads.y)
  );
  kComputeSK<<<
    kComputeSKBlocks, kComputeSKThreads
  >>>(
    num_orbitals, particle, row_stride,
    new_row, inv_det, solution_arr
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto j = 0uz; j < num_orbitals; ++j) {
    if (j == particle) { continue; }

    auto& solution{solution_arr[j]};
    #pragma omp simd reduction(+ : solution)
    for (auto i = 0uz; i < num_orbitals; ++i) {
      stencil::slater::k_compute_sk(
        i, j, row_stride,
        new_row, inv_det,
        &solution
      );
    }
  }
#endif
}

inline void update_trig_cache(
  SlaterPlaneWave::View slater,
  std::size_t particle,
  Particles::View particles
) {
  update_trig_cache(
    slater.num_unique_k,
    particle * slater.trig_row_stride,
    particle,
    particles.pos,
    slater.k_vector,
    slater.sin_cache,
    slater.cos_cache
  );
}

inline void build_trig_cache(
  SlaterPlaneWave::View slater,
  Particles::View particles
) {
  build_trig_cache(
    slater.num_unique_k,
    slater.trig_row_stride,
    particles.pos,
    slater.k_vector,
    slater.sin_cache,
    slater.cos_cache
  );
}

inline void build_determinant(SlaterPlaneWave::View slater) {
  build_determinant(
    slater.num_orbitals,
    slater.trig_row_stride,
    slater.matrix_row_stride,
    slater.sin_cache,
    slater.cos_cache,
    slater.orbital_k_index,
    slater.orbital_type,
    slater.determinant
  );
}

inline void build_row(
  SlaterPlaneWave::View slater,
  std::size_t particle
) {
  build_row(
    slater.num_orbitals,
    particle,
    slater.trig_row_stride,
    slater.sin_cache,
    slater.cos_cache,
    slater.orbital_k_index,
    slater.orbital_type,
    slater.new_row
  );
}

inline void k_compute_sk(
  SlaterPlaneWave::View slater,
  std::size_t particle,
  const fp_t* new_row
) {
  k_compute_sk(
    slater.num_orbitals,
    particle,
    slater.matrix_row_stride,
    new_row,
    slater.inv_determinant,
    slater.solution
  );
}

inline void k_update_inverse(
  SlaterPlaneWave::View slater,
  std::size_t particle,
  fp_t inv_ratio
) {
  k_update_inverse(
    slater.num_orbitals,
    particle,
    slater.matrix_row_stride,
    inv_ratio,
    slater.inv_d_col,
    slater.solution,
    slater.inv_determinant
  );
}

} // namespace kernel::slater
} // namespace kernel
