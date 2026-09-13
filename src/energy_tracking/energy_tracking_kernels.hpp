#pragma once

#include <xpu/xpu.hpp>
#include "energy_tracking.hpp"
#include "../utilities/execution.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

#include "energy_tracking_stencils.hpp"

namespace kernel {
namespace energy {

namespace {

struct KineticEnergyContribution {
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives{
    nullptr, 0uz
  };

  [[nodiscard]] CUDA_CALLABLE
  fp_t operator()(const xpu::array<std::size_t, 1uz>& index) const {
    return stencil::energy::kinetic_energy_contribution(
      index[0uz],
      derivatives
    );
  }
};

#if defined(XPU_CUDA)
__global__
void cudaUpdateRealEnergy(
  std::size_t moved,
  fp_t L, fp_t half_L, fp_t alpha,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* RESTRICT delta
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= pos.count()) { return; }

  xpu::array<fp_t, idx(Axis::NUM)> new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  stencil::energy::update_real_energy(
    i, moved,
    L, half_L, alpha,
    old_pos, new_pos, pos,
    delta
  );
}

__global__
void cudaUpdateStructureFactors(
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::array<fp_t, idx(Axis::NUM)> new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector,
  fp_t* RESTRICT sum_real,
  fp_t* RESTRICT sum_imag
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= g_vector.count()) { return; }

  stencil::energy::update_structure_factors(
    i,
    old_pos, new_pos,
    g_vector,
    sum_real, sum_imag
  );
}
#endif

} // namespace

inline fp_t update_real_energy(
  std::size_t moved,
  fp_t L, fp_t half_L, fp_t alpha,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* RESTRICT delta_scratch
) noexcept {
#if defined(XPU_CUDA)
  xpu::zero_n(delta_scratch, 1uz);

  dim3 updateRealEnergyThreads{256u};
  dim3 updateRealEnergyBlocks{
    xpu::block_per_dim(pos.count(), updateRealEnergyThreads.x)
  };

  cudaUpdateRealEnergy<<<
    updateRealEnergyBlocks, updateRealEnergyThreads
  >>>(
    moved,
    L, half_L, alpha,
    old_pos, pos,
    delta_scratch
  );
  xpu::cu_check(cudaGetLastError());

  auto delta{0.0_fp};
  xpu::copy_n(&delta, delta_scratch, 1uz);
  return delta;
#else
  scast<void>(delta_scratch);

  xpu::array<fp_t, idx(Axis::NUM)> new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  auto delta{0.0_fp};

  #pragma omp simd reduction(+ : delta)
  for (auto i = 0uz; i < pos.count(); ++i) {
    stencil::energy::update_real_energy(
      i, moved,
      L, half_L, alpha,
      old_pos, new_pos, pos,
      &delta
    );
  }

  return delta;
#endif
}

inline std::size_t kinetic_energy_scratch_bytes(
  std::size_t num_particles
) {
  const xpu::range<1uz> range{
    {0uz},
    {num_particles},
    {1uz}
  };
  const KineticEnergyContribution contribution{
    xpu::soa_view<fp_t, idx(Derivatives::NUM)>{nullptr, num_particles}
  };

  return xpu::parallel_reduce_sum_bytes<fp_t>(range, contribution);
}

inline fp_t kinetic_energy(
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives,
  fp_t* RESTRICT kinetic_sum_scratch,
  void* scratch,
  std::size_t scratch_bytes
) noexcept {
  const xpu::range<1uz> range{
    {0uz},
    {derivatives.count()},
    {1uz}
  };
  const KineticEnergyContribution contribution{derivatives};

  xpu::parallel_reduce_sum(
    range,
    kinetic_sum_scratch,
    contribution,
    scratch,
    scratch_bytes
  );

  auto kinetic_sum{0.0_fp};
  xpu::copy_n(&kinetic_sum, kinetic_sum_scratch, 1uz);
  return -0.5_fp * kinetic_sum;
}

inline void update_structure_factors(
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::array<fp_t, idx(Axis::NUM)> new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector,
  fp_t* RESTRICT sum_real,
  fp_t* RESTRICT sum_imag
) noexcept {
#if defined(XPU_CUDA)
  dim3 updateStructureFactorsThreads{256u};
  dim3 updateStructureFactorsBlocks{
    xpu::block_per_dim(
      g_vector.count(), updateStructureFactorsThreads.x
    )
  };

  cudaUpdateStructureFactors<<<
    updateStructureFactorsBlocks, updateStructureFactorsThreads
  >>>(
    old_pos, new_pos,
    g_vector,
    sum_real, sum_imag
  );
  xpu::cu_check(cudaGetLastError());
#else
  #pragma omp simd
  for (auto i = 0uz; i < g_vector.count(); ++i) {
    stencil::energy::update_structure_factors(
      i,
      old_pos, new_pos,
      g_vector,
      sum_real, sum_imag
    );
  }
#endif
}

inline fp_t update_real_energy(
  EnergyTracker::View energy,
  std::size_t moved,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  Particles::View particles
) noexcept {
  return update_real_energy(
    moved,
    energy.box_length,
    0.5_fp * energy.box_length,
    energy.ewald_alpha,
    old_pos,
    particles.pos,
    energy.reduction_scratch
  );
}

inline fp_t kinetic_energy(
  EnergyTracker::View energy,
  Particles::View particles,
  void* scratch,
  std::size_t scratch_bytes
) noexcept {
  return kinetic_energy(
    particles.derivatives,
    energy.reduction_scratch,
    scratch,
    scratch_bytes
  );
}

inline void update_structure_factors(
  EnergyTracker::View energy,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::array<fp_t, idx(Axis::NUM)> new_pos
) noexcept {
  update_structure_factors(
    old_pos,
    new_pos,
    energy.g_vector,
    energy.sum_real,
    energy.sum_imag
  );
}

namespace {

constexpr auto initialization_threads{128uz};
constexpr auto initialization_elements{512uz};

enum class InitializationMode { ALL, STRUCTURE, RECIPROCAL, REAL };

#if defined(XPU_CUDA)
template <typename States>
__global__
void cudaInitializeReciprocalPartials(States states, bool build_structure) {
  const auto partial{scast<std::size_t>(blockIdx.x)};
  const auto walker{scast<std::size_t>(blockIdx.y)};
  const auto state{states[walker]};
  const auto begin{partial * initialization_elements};
  const auto end{xpu::min(begin + initialization_elements, state.energy.num_g_vectors)};
  __shared__ fp_t scratch[initialization_threads];

  stencil::energy::initialize_reciprocal_partial(
    state, begin, end, build_structure, scratch, state.reciprocal_partials + partial
  );
}

template <typename States>
__global__
void cudaInitializeRealPartials(States states) {
  const auto partial{scast<std::size_t>(blockIdx.x)};
  const auto walker{scast<std::size_t>(blockIdx.y)};
  const auto state{states[walker]};
  const auto begin{partial * initialization_elements};
  const auto end{xpu::min(begin + initialization_elements, state.particles.count * state.particles.count)};
  __shared__ fp_t scratch[initialization_threads];

  stencil::energy::initialize_real_partial(state, begin, end, scratch, state.real_partials + partial);
}

template <typename States>
__global__
void cudaFinalizeInitialEnergy(States states, std::size_t reciprocal_count, std::size_t real_count) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  __shared__ fp_t scratch[initialization_threads];
  stencil::energy::finalize_initial_energy(states[walker], reciprocal_count, real_count, scratch);
}
#endif

} // namespace

inline void validate_initialization_size(std::size_t walker_count, std::size_t partial_count) {
#if defined(XPU_CUDA)
  constexpr auto maximum_walker_blocks{65535uz};
  constexpr auto maximum_partial_blocks{2147483647uz};
  if (walker_count > maximum_walker_blocks || partial_count > maximum_partial_blocks) {
    throw std::length_error("Energy initialization exceeds CUDA grid limits");
  }
#else
  scast<void>(walker_count);
  scast<void>(partial_count);
#endif
}

inline std::size_t initialization_partial_count(std::size_t elements) noexcept {
  return xpu::max(1uz, elements / initialization_elements + scast<std::size_t>(elements % initialization_elements != 0uz));
}

namespace {

template <InitializationMode mode, typename States>
inline void initialize(
  States states,
  std::size_t walker_count,
  std::size_t particle_count,
  std::size_t g_count,
  std::size_t num_threads
) noexcept {
  if (walker_count == 0uz) { return; }
  constexpr auto build_structure{mode == InitializationMode::ALL || mode == InitializationMode::STRUCTURE};
  constexpr auto build_reciprocal{mode != InitializationMode::REAL};
  constexpr auto build_real{mode == InitializationMode::ALL || mode == InitializationMode::REAL};
  const auto reciprocal_count{build_reciprocal ? initialization_partial_count(g_count) : 0uz};
  const auto real_count{build_real ? initialization_partial_count(particle_count * particle_count) : 0uz};
  const auto finalize_reciprocal_count{mode == InitializationMode::STRUCTURE ? 0uz : reciprocal_count};

#if defined(XPU_CUDA)
  scast<void>(num_threads);
  if constexpr (build_reciprocal) {
    constexpr dim3 initializeReciprocalPartialsThreads{scast<unsigned int>(initialization_threads)};
    const dim3 initializeReciprocalPartialsBlocks{
      scast<unsigned int>(reciprocal_count), scast<unsigned int>(walker_count)
    };
    cudaInitializeReciprocalPartials<<<
      initializeReciprocalPartialsBlocks, initializeReciprocalPartialsThreads
    >>>(states, build_structure);
    xpu::cu_check(cudaGetLastError());
  }
  if constexpr (build_real) {
    constexpr dim3 initializeRealPartialsThreads{scast<unsigned int>(initialization_threads)};
    const dim3 initializeRealPartialsBlocks{
      scast<unsigned int>(real_count), scast<unsigned int>(walker_count)
    };
    cudaInitializeRealPartials<<<
      initializeRealPartialsBlocks, initializeRealPartialsThreads
    >>>(states);
    xpu::cu_check(cudaGetLastError());
  }
  if constexpr (mode != InitializationMode::STRUCTURE) {
    constexpr dim3 finalizeInitialEnergyThreads{scast<unsigned int>(initialization_threads)};
    const dim3 finalizeInitialEnergyBlocks{scast<unsigned int>(walker_count)};
    cudaFinalizeInitialEnergy<<<
      finalizeInitialEnergyBlocks, finalizeInitialEnergyThreads
    >>>(states, finalize_reciprocal_count, real_count);
    xpu::cu_check(cudaGetLastError());
  }
#else
  scast<void>(num_threads);
  #pragma omp parallel for num_threads(num_threads)
  for (auto walker = 0uz; walker < walker_count; ++walker) {
    const auto state{states[walker]};
    auto scratch{0.0_fp};
    for (auto partial{0uz}; partial < reciprocal_count; ++partial) {
      const auto begin{partial * initialization_elements};
      const auto end{xpu::min(begin + initialization_elements, g_count)};
      stencil::energy::initialize_reciprocal_partial(
        state, begin, end, build_structure, &scratch, state.reciprocal_partials + partial
      );
    }
    for (auto partial{0uz}; partial < real_count; ++partial) {
      const auto begin{partial * initialization_elements};
      const auto end{xpu::min(begin + initialization_elements, particle_count * particle_count)};
      stencil::energy::initialize_real_partial(state, begin, end, &scratch, state.real_partials + partial);
    }
    stencil::energy::finalize_initial_energy(state, finalize_reciprocal_count, real_count, &scratch);
  }
#endif
}

} // namespace

inline void initialize(
  EnergyTracker::InitializationBatchView states,
  std::size_t num_threads
) noexcept {
  initialize<InitializationMode::ALL>(
    states,
    states.energy.walker_count(),
    states.particles.count(),
    states.energy.num_g_vectors,
    num_threads
  );
}

inline void initialize_reciprocal_energy(EnergyTracker::InitializationView state) noexcept {
  const xpu::array<EnergyTracker::InitializationView, 1uz> states{state};
  initialize<InitializationMode::RECIPROCAL>(states, 1uz, state.particles.count, state.energy.num_g_vectors, 1uz);
}

inline void initialize_real_energy(EnergyTracker::InitializationView state) noexcept {
  const xpu::array<EnergyTracker::InitializationView, 1uz> states{state};
  initialize<InitializationMode::REAL>(states, 1uz, state.particles.count, state.energy.num_g_vectors, 1uz);
}

inline void initialize_structure_factors(EnergyTracker::InitializationView state) noexcept {
  const xpu::array<EnergyTracker::InitializationView, 1uz> states{state};
  initialize<InitializationMode::STRUCTURE>(states, 1uz, state.particles.count, state.energy.num_g_vectors, 1uz);
}

namespace {

#if defined(XPU_CUDA)
__global__
void cudaAcceptEnergyMove(EnergyTracker::View energy, fp_t real_delta, fp_t reciprocal, bool update_reciprocal) {
  stencil::energy::accept_move(energy, real_delta, reciprocal, update_reciprocal);
}
#endif

} // namespace

inline void accept_move(EnergyTracker::View energy, fp_t real_delta, fp_t reciprocal, bool update_reciprocal) noexcept {
#if defined(XPU_CUDA)
  constexpr dim3 acceptEnergyMoveThreads{1u};
  constexpr dim3 acceptEnergyMoveBlocks{1u};
  cudaAcceptEnergyMove<<<
    acceptEnergyMoveBlocks, acceptEnergyMoveThreads
  >>>(energy, real_delta, reciprocal, update_reciprocal);
  xpu::cu_check(cudaGetLastError());
#else
  stencil::energy::accept_move(energy, real_delta, reciprocal, update_reciprocal);
#endif
}

} // namespace kernel::energy
} // namespace kernel
