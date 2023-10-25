/*
 * ----- CLEO -----
 * File: condensation.cpp
 * Project: superdrops
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Wednesday 25th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * Functionality related to modelling condensation
 * microphysical process in SDM
 */

#include "./condensation.hpp"

KOKKOS_FUNCTION
subviewd_supers DoCondensation::do_condensation(const unsigned int subt,
                                     const subviewd_supers supers) const
/* enact condensation / evaporation microphysical process */
{
  return supers;
}