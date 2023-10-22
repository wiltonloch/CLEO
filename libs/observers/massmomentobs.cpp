/*
 * ----- CLEO -----
 * File: massmomentobs.cpp
 * Project: observers
 * Created Date: Sunday 22nd October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Sunday 22nd October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * functionality to calculate and output mass moments 
 * of droplet distribution to array in a zarr file
 * system storage
 */

#include "./massmomentobs.hpp"

void DoMassMomentsObs::
    massmoments_to_storage(const mirrorh_constsupers h_supers)
/* calculated 0th, 1st and 2nd moment of the (real) droplet mass
distribution and then writes them to zarr storage. (I.e.
0th, 3rd and 6th moment of the droplet radius distribution) */
{
  double mom0(0.0); // 0th moment = number of (real) droplets
  double mom1(0.0); // 1st moment = mass of (real) droplets
  double mom2(0.0); // 2nd moment = mass^2 of (real) droplets
  for (size_t kk(0); kk < h_supers.extent(0); ++kk)
  {
    const(double) xi(h_supers(kk).get_xi()); // cast multiplicity from unsigned int to double
    const double mass(h_supers(kk).mass());
    mom0 += xi;
    mom1 += xi * mass;
    mom2 += xi * mass * mass;
  }

  zarr->values_to_storage(mom0, mom1, mom2);
}
