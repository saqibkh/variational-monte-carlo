#pragma once

#include <xpu/soa.hpp>
#include "../utilities/macros.hpp"
#include "../utilities/components.hpp"

#include <cstdlib>
#include <cstring>

class Particles {
private:
  enum class ArrayIndex {
    POS,
    DERIVATIVES = enum_index(Axis::NUM, POS),
    NUM_SUB_ARRAYS = enum_index(Derivatives::NUM, DERIVATIVES)
  };
  xpu::soa_batch<fp_t, idx(ArrayIndex::NUM_SUB_ARRAYS)> data_;

public:
  struct View {
    std::size_t count{};
    vec3_view_t pos{nullptr, 0uz};
    derivatives_view derivatives{nullptr, 0uz};
  };

  struct BatchView {
    vec3_batch_view_t pos{
      nullptr, 0uz, 0uz, 0uz
    };
    derivatives_batch_view derivatives{
      nullptr, 0uz, 0uz, 0uz
    };

    [[nodiscard]] CUDA_CALLABLE
    std::size_t count() const noexcept {
      return pos.element_count();
    }

    [[nodiscard]] CUDA_CALLABLE
    std::size_t walker_count() const noexcept {
      return pos.batch_count();
    }

    [[nodiscard]] CUDA_CALLABLE
    View view(std::size_t walker) noexcept {
      return {
        this->count(),
        pos.view(walker),
        derivatives.view(walker)
      };
    }
  };

  explicit Particles(
    std::size_t num_particles,
    std::size_t num_walkers = 1uz
  )
    : data_{num_walkers, num_particles}
  { }

  [[nodiscard]]
  std::size_t count() const {
    return data_.element_count();
  }

  [[nodiscard]]
  std::size_t walker_count() const {
    return data_.batch_count();
  }

  [[nodiscard]]
  std::size_t stride() const {
    return data_.array_stride();
  }

  [[nodiscard]]
  vec3_view_t pos(std::size_t walker = 0uz) {
    return data_.view<idx(Axis::NUM), idx(ArrayIndex::POS)>(walker);
  }
  
  [[nodiscard]]
  const_vec3_view_t pos(std::size_t walker = 0uz) const {
    return data_.view<idx(Axis::NUM), idx(ArrayIndex::POS)>(walker);
  }

  [[nodiscard]]
  derivatives_view derivatives(std::size_t walker = 0uz) {
    return data_.view<idx(Derivatives::NUM), idx(ArrayIndex::DERIVATIVES)>(walker);
  }
  
  [[nodiscard]]
  const_derivatives_view derivatives(std::size_t walker = 0uz) const {
    return data_.view<idx(Derivatives::NUM), idx(ArrayIndex::DERIVATIVES)>(walker);
  }

  [[nodiscard]]
  BatchView batch_view() noexcept {
    return {
      data_.view<idx(Axis::NUM), idx(ArrayIndex::POS)>(),
      data_.view<idx(Derivatives::NUM), idx(ArrayIndex::DERIVATIVES)>()
    };
  }

  [[nodiscard]]
  View view(std::size_t walker = 0uz) noexcept {
    return this->batch_view().view(walker);
  }
};
