/*
 * ----- CLEO -----
 * File: gridboxmaps.hpp
 * Project: gridboxes
 * Created Date: Tuesday 17th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Tuesday 17th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * concept for maps to convert between a gridbox indexes
 * and domain coordinates for a and type of C grid used
 * by CLEO SDM
 */


#ifndef GRIDBOXMAPS_HPP
#define GRIDBOXMAPS_HPP

#include <concepts>

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

template <typename GbxMaps>
concept GridboxMaps = requires(GbxMaps gbxmaps, unsigned int ii)
/* concept Observer is all types that have an operator that
has signature of observing functions (see Observer concept) */
{
  {
    gbxmaps.coord3bounds(ii)
  } -> std::convertible_to<Kokkos::pair<double, double>>;
  {
    gbxmaps.coord1bounds(ii)
  } -> std::convertible_to<Kokkos::pair<double, double>>;
  {
    gbxmaps.coord2bounds(ii)
  } -> std::convertible_to<Kokkos::pair<double, double>>;
};

#endif // GRIDBOXMAPS_HPP