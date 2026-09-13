#pragma once

#include <xpu/xpu.hpp>
#include <xpu/random.hpp>
#include "../utilities/execution.hpp"
#include "../utilities/macros.hpp"
#include "simulation.hpp"
#include "simulation_stencils.hpp"

#include <cstddef>
#include <cstdint>

namespace kernel {
namespace simulation {

#if defined(XPU_CUDA)

namespace {

__global__
void cudaSeedGenerator(
  xpu::random::generator* generator,
  std::uint64_t master_seed,
  std::uint64_t walker_id
) {
  stencil::simulation::seed_generator(generator, master_seed, walker_id);
}

__global__
void cudaSeedGenerators(
  xpu::random::generator* generators,
  std::size_t walker_count,
  std::uint64_t master_seed,
  std::uint64_t first_walker_id
) {
  const auto [walker]{xpu::global_index<1>()};
  if (walker >= walker_count) { return; }

  stencil::simulation::seed_generator(
    generators + walker,
    master_seed,
    first_walker_id + walker
  );
}

__global__
void cudaInitializeWalkerPositions(
  const Simulation::View simulation,
  fp_t box_length
) {
  stencil::simulation::initialize_walker_positions(
    simulation,
    box_length
  );
}

__global__
void cudaInitializePositions(
  Simulation::BatchView simulations,
  fp_t box_length
) {
  const auto [walker]{xpu::global_index<1>()};
  if (walker >= simulations.walker_count()) { return; }

  const auto simulation{simulations.view(walker)};
  stencil::simulation::initialize_walker_positions(
    simulation,
    box_length
  );
}

__global__
void cudaMetropolisStep(
  Simulation::View simulation,
  fp_t step_size
) {
  __shared__ Simulation::MetropolisScratch scratch;

  stencil::simulation::metropolis_step(
    simulation,
    step_size,
    scratch
  );

  if (execution::thread() == 0uz) {
    *simulation.step_result = scratch.result;
  }
}

__global__
void cudaMetropolisSweep(
  Simulation::BatchView simulations,
  std::size_t proposals_per_walker,
  fp_t step_size,
  Simulation::SweepResult* result
) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  if (walker >= simulations.walker_count()) { return; }

  const auto simulation{simulations.view(walker)};

  __shared__ Simulation::MetropolisScratch scratch;
  __shared__ Simulation::SweepResult local_result;

  if (execution::thread() == 0uz) {
    local_result = {};
  }
  execution::sync();

  for (auto proposal{0uz}; proposal < proposals_per_walker; ++proposal) {
    stencil::simulation::metropolis_step(
      simulation,
      step_size,
      scratch
    );

    if (execution::thread() == 0uz) {
      ++local_result.proposed;
      local_result.accepted += scast<std::size_t>(scratch.result.accepted);
    }
    execution::sync();
  }

  if (execution::thread() == 0uz) {
    static_assert(sizeof(std::size_t) == sizeof(unsigned long long));
    atomicAdd(
      reinterpret_cast<unsigned long long*>(&result->proposed),
      scast<unsigned long long>(local_result.proposed)
    );
    atomicAdd(
      reinterpret_cast<unsigned long long*>(&result->accepted),
      scast<unsigned long long>(local_result.accepted)
    );
  }
}

__global__
void cudaMeasureWalkers(
  Simulation::BatchView simulations
) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  if (walker >= simulations.walker_count()) { return; }

  const auto simulation{simulations.view(walker)};
  stencil::simulation::measure_walker(simulation);
}

__global__
void cudaRunWalkers(
  Simulation::BatchView simulations,
  Simulation::WalkerState* states,
  Simulation::RunConfig config
) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  if (walker >= simulations.walker_count()) {
    return;
  }

  const auto simulation{simulations.view(walker)};
  __shared__ Simulation::MetropolisScratch scratch;

  stencil::simulation::run_walker(
    simulation,
    states[walker],
    config,
    scratch
  );
}

__global__
void cudaFinalizeRun(
  const Simulation::WalkerState* states,
  std::size_t walker_count,
  Simulation::RunResult* result
) {
  if (execution::thread() != 0uz || blockIdx.x != 0u) {
    return;
  }

  stencil::simulation::finalize_run(
    states,
    walker_count,
    result
  );
}

} // namespace

#endif

inline void seed_generator(
  xpu::random::generator* generator,
  std::uint64_t master_seed,
  std::uint64_t walker_id
) {
#if defined(XPU_CUDA)
  constexpr dim3 seedGeneratorThreads{1u};
  constexpr dim3 seedGeneratorBlocks{1u};

  cudaSeedGenerator<<<
    seedGeneratorBlocks, seedGeneratorThreads
  >>>(
    generator,
    master_seed,
    walker_id
  );
  xpu::cu_check(cudaGetLastError());
#else
  stencil::simulation::seed_generator(generator, master_seed, walker_id);
#endif
}

inline void seed_generators(
  xpu::random::generator* generators,
  std::size_t walker_count,
  std::uint64_t master_seed,
  std::uint64_t first_walker_id
) {
  if (walker_count == 0uz) { return; }

#if defined(XPU_CUDA)
  constexpr dim3 seedGeneratorsThreads{256u};
  const dim3 seedGeneratorsBlocks{
    xpu::block_per_dim(walker_count, seedGeneratorsThreads.x)
  };

  cudaSeedGenerators<<<
    seedGeneratorsBlocks, seedGeneratorsThreads
  >>>(
    generators,
    walker_count,
    master_seed,
    first_walker_id
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    stencil::simulation::seed_generator(
      generators + walker,
      master_seed,
      first_walker_id + walker
    );
  }
#endif
}

inline void initialize_positions(
  const Simulation::View simulation,
  fp_t box_length
) {
#if defined(XPU_CUDA)
  constexpr dim3 initializePositionsThreads{1u};
  constexpr dim3 initializePositionsBlocks{1u};

  cudaInitializeWalkerPositions<<<
    initializePositionsBlocks, initializePositionsThreads
  >>>(
    simulation,
    box_length
  );
  xpu::cu_check(cudaGetLastError());
#else
  stencil::simulation::initialize_walker_positions(
    simulation,
    box_length
  );
#endif
}

inline void initialize_positions(
  Simulation::BatchView simulations,
  fp_t box_length
) {
  const auto walker_count{simulations.walker_count()};

#if defined(XPU_CUDA)
  constexpr dim3 initializePositionsThreads{256u};
  const dim3 initializePositionsBlocks{
    xpu::block_per_dim(walker_count, initializePositionsThreads.x)
  };

  cudaInitializePositions<<<
    initializePositionsBlocks, initializePositionsThreads
  >>>(
    simulations,
    box_length
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    stencil::simulation::initialize_walker_positions(
      simulation,
      box_length
    );
  }
#endif
}

inline Simulation::StepResult metropolis_step(
  Simulation::View simulation,
  fp_t step_size
) {
#if defined(XPU_CUDA)
  dim3 metropolisStepThreads{256u};
  dim3 metropolisStepBlocks{1u};
  cudaMetropolisStep<<<
    metropolisStepBlocks, metropolisStepThreads
  >>>(
    simulation,
    step_size
  );
  xpu::cu_check(cudaGetLastError());

  Simulation::StepResult result{};
  xpu::copy_n(&result, simulation.step_result, 1uz);
  return result;
#else
  Simulation::MetropolisScratch scratch{};
  stencil::simulation::metropolis_step(
    simulation,
    step_size,
    scratch
  );
  *simulation.step_result = scratch.result;
  return scratch.result;
#endif
}

inline void metropolis_sweep(
  Simulation::BatchView simulations,
  std::size_t proposals_per_walker,
  fp_t step_size,
  Simulation::SweepResult* result
) {
  const auto walker_count{simulations.walker_count()};
  xpu::memset(result, 0, sizeof(Simulation::SweepResult));

#if defined(XPU_CUDA)
  dim3 metropolisSweepThreads{256u};
  dim3 metropolisSweepBlocks{scast<unsigned int>(walker_count)};
  cudaMetropolisSweep<<<
    metropolisSweepBlocks, metropolisSweepThreads
  >>>(
    simulations,
    proposals_per_walker,
    step_size,
    result
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    Simulation::SweepResult local_result{};

    for (auto proposal{0uz}; proposal < proposals_per_walker; ++proposal) {
      Simulation::MetropolisScratch scratch{};
      stencil::simulation::metropolis_step(
        simulation,
        step_size,
        scratch
      );

      ++local_result.proposed;
      local_result.accepted += scast<std::size_t>(scratch.result.accepted);
    }

    result->proposed += local_result.proposed;
    result->accepted += local_result.accepted;
  }
#endif
}

inline void measure_walkers(
  Simulation::BatchView simulations
) {
  const auto walker_count{simulations.walker_count()};

#if defined(XPU_CUDA)
  dim3 measureWalkersThreads{256u};
  dim3 measureWalkersBlocks{scast<unsigned int>(walker_count)};
  cudaMeasureWalkers<<<
    measureWalkersBlocks, measureWalkersThreads
  >>>(
    simulations
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    stencil::simulation::measure_walker(simulation);
  }
#endif
}

inline Simulation::RunResult run_walkers(
  Simulation::BatchView simulations,
  Simulation::WalkerState* states,
  const Simulation::RunConfig& config,
  Simulation::RunResult* result_storage
) {
  const auto walker_count{simulations.walker_count()};

#if defined(XPU_CUDA)
  constexpr dim3 runWalkersThreads{256u};
  const dim3 runWalkersBlocks{scast<unsigned int>(walker_count)};

  cudaRunWalkers<<<
    runWalkersBlocks, runWalkersThreads
  >>>(
    simulations,
    states,
    config
  );
  xpu::cu_check(cudaGetLastError());

  constexpr dim3 finalizeRunThreads{1u};
  constexpr dim3 finalizeRunBlocks{1u};

  cudaFinalizeRun<<<
    finalizeRunBlocks, finalizeRunThreads
  >>>(
    states,
    walker_count,
    result_storage
  );
  xpu::cu_check(cudaGetLastError());
#else
  #pragma omp parallel for num_threads(config.num_threads)
  for (auto walker = 0uz; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    Simulation::MetropolisScratch scratch{};

    stencil::simulation::run_walker(
      simulation,
      states[walker],
      config,
      scratch
    );
  }

  stencil::simulation::finalize_run(
    states,
    walker_count,
    result_storage
  );
#endif

  Simulation::RunResult result{};
  xpu::copy_n(&result, result_storage, 1uz);
  return result;
}

} // namespace kernel::simulation
} // namespace kernel
