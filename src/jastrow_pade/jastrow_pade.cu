#include <xpu/xpu.hpp>
#include "jastrow_pade.hpp"
#include "jastrow_pade_kernels.hpp"

#include <cstddef>

void JastrowPade::update_derivatives_for_move(
  std::size_t moved,
  vec3_t old_pos,
  Particles::View particles,
  derivatives_view derivatives
) const noexcept {
  kernel::jpade::compute_derivatives(
    this->view(), moved, old_pos, particles, derivatives
  );
}

void JastrowPade::add_derivatives(
  Particles::View particles,
  derivatives_view derivatives
) const noexcept {
  kernel::jpade::add_derivatives(this->view(), particles, derivatives);
}

fp_t JastrowPade::value(
  Particles::View particles
) const noexcept {
  return kernel::jpade::value(this->view(), particles);
}

fp_t JastrowPade::delta_value(
  std::size_t moved,
  vec3_t old_pos,
  Particles::View particles
) const noexcept {
  return kernel::jpade::delta_value(this->view(), moved, old_pos, particles);
}
