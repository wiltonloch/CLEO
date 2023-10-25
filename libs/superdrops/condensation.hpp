/*
 * ----- CLEO -----
 * File: condensation.hpp
 * Project: superdrops
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 26th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * struct for condensation / evaporation of water
 * causing diffusional growth / shrinking of
 * droplets in SDM. Equations referenced as (eqn [X.YY])
 * are from "An Introduction To Clouds From The 
 * Microscale to Climate" by Lohmann, Luond
 * and Mahrt, 1st edition.
 */

#ifndef CONDENSATION_HPP
#define CONDENSATION_HPP

#include <concepts>

#include <Kokkos_Core.hpp>

#include "../cleoconstants.hpp"
#include "./kokkosaliases_sd.hpp"
#include "./microphysicalprocess.hpp"
#include "./superdrop.hpp"
#include "./state.hpp"
#include "./thermodynamic_equations.hpp"
#include "./urbg.hpp"

struct DoCondensation
/* function-like type that enacts
condensation / evaporation microphysical process */
{
private:

  KOKKOS_FUNCTION
  void do_condensation(const subviewd_supers supers, State &state) const;
  /* Enacts condensation / evaporation microphysical process.
  Change to superdroplet radii and temp, qv and qc due to
  sum of radii changes via diffusion and condensation of
  water vapour during timestep delt. Using equations
  from "An Introduction To Clouds...." (see note at top of file) */

  KOKKOS_FUNCTION
  double condensation_mass_change(Superdrop &drop,
                                  const double press,
                                  const double temp,
                                  const double psat,
                                  const double s_ratio) const;

public:
  template <class DeviceType>
  KOKKOS_INLINE_FUNCTION
  subviewd_supers operator()(const unsigned int subt,
                             subviewd_supers supers,
                             State &state,
                             URBG<DeviceType> urbg) const
  /* this operator is used as an "adaptor" for using
  condensation as the MicrophysicsFunction type in a
  ConstTstepMicrophysics instance (*hint* which itself
  satsifies the MicrophysicalProcess concept) */
  {
    do_condensation(supers, state);

    return supers;
  }
};

inline MicrophysicalProcess auto
Condensation(const unsigned int interval)
/* constructs Microphysical Process for
condensation/evaporation of superdroplets with a
constant timestep 'interval' given the
"do_condensation" function-like type */
{
  return ConstTstepMicrophysics(interval, DoCondensation{});
}

/* -----  ----- TODO: move functions below to .cpp file ----- ----- */

KOKKOS_FUNCTION
void DoCondensation::do_condensation(const subviewd_supers supers,
                                     State &state) const
/* Enacts condensation / evaporation microphysical process.
Change to superdroplet radii and temp, qv and qc due to
sum of radii changes via diffusion and condensation of
water vapour during timestep delt. Using equations
from "An Introduction To Clouds...." (see note at top of file) */
{
  /* superdroplet radii changes */
  constexpr double C0cubed(dlc::COORD0 * dlc::COORD0 * dlc::COORD0);
  const double VOLUME(state.get_volume() * C0cubed);    // volume in which condensation occurs [m^3]
  const double psat(saturation_pressure(state.temp));
  const double s_ratio(supersaturation_ratio(state.press, state.qvap, psat));

  double totmass_condensed(0.0); // cumulative change to liquid mass in parcel volume 'dm'
  for (size_t kk(0); kk < supers.extent(0); ++kk)
  {
    const double deltamass_condensed(
        condensation_mass_change(supers(kk), state.press,
                                 state.temp, psat, s_ratio));
    totmass_condensed += deltamass_condensed; // dm += dm_condensed_vapour/dt * delta t
  }
  const double totrho_condensed(totmass_condensed / VOLUME); // drho_condensed_vapour/dt * delta t

  // /* resultant effect on thermodynamic state */
  // if (doAlterThermo)
  // {
  //   condensation_alters_thermostate(state, totrho_condensed); // TODO
  // }
}

KOKKOS_FUNCTION
double DoCondensation::condensation_mass_change(Superdrop &drop,
                                                const double press,
                                                const double temp,
                                                const double psat,
                                                const double s_ratio) const
/* update superdroplet radius due to radial growth/shrink
  via condensation and diffusion of water vapour according
  to equations from "An Introduction To Clouds...." (see
  note at top of file). Then return mass of liquid that
  condensed onto /evaporated off of droplet. New radius is
  calculated using impliciteuler method which iterates
  condensation-diffusion ODE given the previous radius. */
{
  constexpr double R0cubed = dlc::R0 * dlc::R0 * dlc::R0;
  constexpr double dmdt_const = 4.0 * M_PI * dlc::Rho_l * R0cubed;

  const auto fkl_fdl = diffusion_factors(press, temp, psat); // pair = {fkl, fdl}
  const double fkl(fkl_fdl.first);
  const double fdl(fkl_fdl.second);
  const double akoh(akohler_factor(temp));
  const double bkoh(bkohler_factor(drop));

  /* do not pass r by reference here!! copy value into iterator */
  // const double newradius = impliciteuler.solve_condensation(s_ratio,
  //                                                     akoh, bkoh, fkl,
  //                                                     fdl, drop.radius); // timestepping eqn [7.28] forward
  // const double delta_radius = drop.change_radius(newradius);

  const double delta_radius(1.0);
  const double rsqrd(drop.get_radius() * drop.get_radius());
  const double mass_condensed = (dmdt_const * rsqrd * drop.get_xi() * delta_radius); // eqn [7.22] * delta t

  return mass_condensed;
}

#endif // CONDENSATION_HPP