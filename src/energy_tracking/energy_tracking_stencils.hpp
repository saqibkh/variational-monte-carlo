#pragma once

#include <xpu/xpu.hpp>
#include "energy_tracking.hpp"
#include "../utilities/execution.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

#if defined(XPU_CUDA)
  #include <cuda/std/numbers>
#else
  #include <numbers>
#endif

namespace stencil {
namespace energy {

namespace {

CUDA_CALLABLE
inline fp_t evaluate_pair_energy(
  std::size_t other,
  fp_t L, fp_t half_L, fp_t alpha,
  const xpu::array<fp_t, idx(Axis::NUM)>& particle_pos,
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
  const auto inverse_distance{
    (distance < 1e-12_fp) ? 1.0_fp : 1.0_fp / distance
  };

  return xpu::erfc(alpha * distance) * inverse_distance;
}

} // namespace

CUDA_CALLABLE
inline void initialize_reciprocal_energy(
  std::size_t i,
  const fp_t* RESTRICT g_weights,
  const fp_t* RESTRICT sum_real,
  const fp_t* RESTRICT sum_imag,
  fp_t* RESTRICT reciprocal_sum
) noexcept {
  const auto contribution{
    g_weights[i] * (
      sum_real[i] * sum_real[i] +
      sum_imag[i] * sum_imag[i]
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(reciprocal_sum, contribution);
#else
  *reciprocal_sum += contribution;
#endif
}

CUDA_CALLABLE
inline void update_real_energy(
  std::size_t other, std::size_t moved,
  fp_t L, fp_t half_L, fp_t alpha,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* RESTRICT delta
) noexcept {
  if (other == moved) { return; }

  const auto old_pair_energy{
    evaluate_pair_energy(
      other,
      L, half_L, alpha,
      old_pos, pos
    )
  };
  const auto new_pair_energy{
    evaluate_pair_energy(
      other,
      L, half_L, alpha,
      new_pos, pos
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(delta, new_pair_energy - old_pair_energy);
#else
  *delta += new_pair_energy - old_pair_energy;
#endif
}

CUDA_CALLABLE
inline fp_t kinetic_energy_contribution(
  std::size_t i,
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives
) noexcept {
  const auto gradient_x{derivatives[idx(Derivatives::GRAD_X)][i]};
  const auto gradient_y{derivatives[idx(Derivatives::GRAD_Y)][i]};
  const auto gradient_z{derivatives[idx(Derivatives::GRAD_Z)][i]};
  const auto gradient_squared{
    gradient_x * gradient_x +
    gradient_y * gradient_y +
    gradient_z * gradient_z
  };

  return derivatives[idx(Derivatives::LAP)][i] + gradient_squared;
}

CUDA_CALLABLE
inline void kinetic_energy(
  std::size_t i,
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives,
  fp_t* RESTRICT kinetic_sum
) noexcept {
  const auto contribution{kinetic_energy_contribution(i, derivatives)};

#if defined(__CUDA_ARCH__)
  atomicAdd(kinetic_sum, contribution);
#else
  *kinetic_sum += contribution;
#endif
}

CUDA_CALLABLE
inline void kinetic_energy(
  std::size_t particle,
  EnergyTracker::View energy,
  Particles::View particles
) noexcept;

CUDA_CALLABLE
inline void evaluate_local_energy(
  EnergyTracker::View energy,
  Particles::View particles,
  fp_t* local_energy
) noexcept {
  if (execution::thread() == 0uz) {
    *energy.reduction_scratch = 0.0_fp;
  }
  execution::sync();

  for (auto particle{execution::thread()}; particle < particles.count; particle += execution::stride()) {
    kinetic_energy(particle, energy, particles);
  }
  execution::sync();

  if (execution::thread() == 0uz) {
    *local_energy =
      -0.5_fp * *energy.reduction_scratch
      + *energy.real_energy
      + *energy.reciprocal_energy
      + energy.ewald_correction
      + energy.ewald_background;
  }
  execution::sync();
}

CUDA_CALLABLE
inline void update_structure_factors(
  std::size_t i,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector,
  fp_t* RESTRICT sum_real,
  fp_t* RESTRICT sum_imag
) noexcept {
  const auto old_dot{
    g_vector[idx(Axis::X)][i] * old_pos[idx(Axis::X)] +
    g_vector[idx(Axis::Y)][i] * old_pos[idx(Axis::Y)] +
    g_vector[idx(Axis::Z)][i] * old_pos[idx(Axis::Z)]
  };
  const auto new_dot{
    g_vector[idx(Axis::X)][i] * new_pos[idx(Axis::X)] +
    g_vector[idx(Axis::Y)][i] * new_pos[idx(Axis::Y)] +
    g_vector[idx(Axis::Z)][i] * new_pos[idx(Axis::Z)]
  };

  auto old_sin{0.0_fp};
  auto old_cos{0.0_fp};
  auto new_sin{0.0_fp};
  auto new_cos{0.0_fp};

  xpu::sincos(old_dot, &old_sin, &old_cos);
  xpu::sincos(new_dot, &new_sin, &new_cos);

  sum_real[i] += new_cos - old_cos;
  sum_imag[i] += new_sin - old_sin;
}

CUDA_CALLABLE
inline void initialize_reciprocal_energy(
  std::size_t g_index,
  EnergyTracker::View energy,
  fp_t* reciprocal_sum
) noexcept {
  initialize_reciprocal_energy(
    g_index,
    energy.g_weights,
    energy.sum_real,
    energy.sum_imag,
    reciprocal_sum
  );
}

CUDA_CALLABLE
inline void update_structure_factors(
  std::size_t g_index,
  EnergyTracker::View energy,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos
) noexcept {
  update_structure_factors(
    g_index,
    old_pos,
    new_pos,
    energy.g_vector,
    energy.sum_real,
    energy.sum_imag
  );
}

CUDA_CALLABLE
inline void update_real_energy(
  std::size_t other,
  std::size_t moved,
  EnergyTracker::View energy,
  Particles::View particles,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  fp_t* delta
) noexcept {
  update_real_energy(
    other,
    moved,
    energy.box_length,
    0.5_fp * energy.box_length,
    energy.ewald_alpha,
    old_pos,
    new_pos,
    particles.pos,
    delta
  );
}

CUDA_CALLABLE
inline void kinetic_energy(
  std::size_t particle,
  EnergyTracker::View energy,
  Particles::View particles
) noexcept {
  kinetic_energy(
    particle,
    particles.derivatives,
    energy.reduction_scratch
  );
}

CUDA_CALLABLE
inline fp_t reciprocal_contribution(std::size_t g, EnergyTracker::View energy) noexcept {
  return energy.g_weights[g] * (
    energy.sum_real[g] * energy.sum_real[g] +
    energy.sum_imag[g] * energy.sum_imag[g]
  );
}

CUDA_CALLABLE
inline void initialize_structure_factor(
  std::size_t g,
  EnergyTracker::View energy,
  Particles::View particles
) noexcept {
  const auto g_x{energy.g_vector[idx(Axis::X)][g]};
  const auto g_y{energy.g_vector[idx(Axis::Y)][g]};
  const auto g_z{energy.g_vector[idx(Axis::Z)][g]};
  auto real_sum{0.0_fp};
  auto imag_sum{0.0_fp};

  for (auto particle{0uz}; particle < particles.count; ++particle) {
    const auto phase{
      g_x * particles.pos[idx(Axis::X)][particle] +
      g_y * particles.pos[idx(Axis::Y)][particle] +
      g_z * particles.pos[idx(Axis::Z)][particle]
    };
    auto sine{0.0_fp};
    auto cosine{0.0_fp};
    xpu::sincos(phase, &sine, &cosine);
    real_sum += cosine;
    imag_sum += sine;
  }

  energy.sum_real[g] = real_sum;
  energy.sum_imag[g] = imag_sum;
}

CUDA_CALLABLE
inline void store_partial_sum(fp_t contribution, fp_t* scratch, fp_t* result) noexcept {
  const auto thread{execution::thread()};
  scratch[thread] = contribution;
  execution::sync();

  for (auto offset{execution::stride() / 2uz}; offset > 0uz; offset /= 2uz) {
    if (thread < offset) {
      scratch[thread] += scratch[thread + offset];
    }
    execution::sync();
  }

  if (thread == 0uz) {
    *result = scratch[0uz];
  }
  execution::sync();
}

CUDA_CALLABLE
inline void initialize_reciprocal_partial(
  EnergyTracker::InitializationView state,
  std::size_t begin,
  std::size_t end,
  bool build_structure,
  fp_t* scratch,
  fp_t* partial_sum
) noexcept {
  auto contribution{0.0_fp};
  for (auto g{begin + execution::thread()}; g < end; g += execution::stride()) {
    if (build_structure) {
      initialize_structure_factor(g, state.energy, state.particles);
    }
    contribution += reciprocal_contribution(g, state.energy);
  }
  store_partial_sum(contribution, scratch, partial_sum);
}

CUDA_CALLABLE
inline void initialize_real_partial(
  EnergyTracker::InitializationView state,
  std::size_t begin,
  std::size_t end,
  fp_t* scratch,
  fp_t* partial_sum
) noexcept {
  const auto energy{state.energy};
  const auto particles{state.particles};
  auto contribution{0.0_fp};

  for (auto element{begin + execution::thread()}; element < end; element += execution::stride()) {
    const auto first{element / particles.count};
    const auto second{element - first * particles.count};
    if (first >= second) { continue; }

    const xpu::array<fp_t, idx(Axis::NUM)> particle_pos{
      particles.pos[idx(Axis::X)][first],
      particles.pos[idx(Axis::Y)][first],
      particles.pos[idx(Axis::Z)][first]
    };
    contribution += evaluate_pair_energy(
      second,
      energy.box_length, 0.5_fp * energy.box_length, energy.ewald_alpha,
      particle_pos, particles.pos
    );
  }
  store_partial_sum(contribution, scratch, partial_sum);
}

CUDA_CALLABLE
inline void finalize_initial_energy(
  EnergyTracker::InitializationView state,
  std::size_t reciprocal_partial_count,
  std::size_t real_partial_count,
  fp_t* scratch
) noexcept {
  if (reciprocal_partial_count > 0uz) {
    auto sum{0.0_fp};
    for (auto partial{execution::thread()}; partial < reciprocal_partial_count; partial += execution::stride()) {
      sum += state.reciprocal_partials[partial];
    }
    const auto length{state.energy.box_length};
    const auto prefactor{1.0_fp / (2.0_fp * xstd::numbers::pi_v<fp_t> * length * length * length)};
    store_partial_sum(prefactor * sum, scratch, state.energy.reciprocal_energy);
  }
  if (real_partial_count > 0uz) {
    auto sum{0.0_fp};
    for (auto partial{execution::thread()}; partial < real_partial_count; partial += execution::stride()) {
      sum += state.real_partials[partial];
    }
    store_partial_sum(sum, scratch, state.energy.real_energy);
  }
}

CUDA_CALLABLE
inline void accept_move(
  EnergyTracker::View energy,
  fp_t real_delta,
  fp_t reciprocal,
  bool update_reciprocal
) noexcept {
  *energy.real_energy += real_delta;
  if (update_reciprocal) {
    *energy.reciprocal_energy = reciprocal;
  }
}

} // namespace stencil::energy
} // namespace stencil
