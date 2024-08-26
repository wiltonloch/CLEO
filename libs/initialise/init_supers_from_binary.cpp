/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: init_supers_from_binary.cpp
 * Project: initialise
 * Created Date: Monday 30th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Friday 19th April 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * struct for reading in some super-droplets' initial conditions for CLEO SDM
 * (e.g. superdroplet attributes) from a binary file. InitAllSupersFromBinary instance
 * can be used by InitConds struct as SuperdropInitConds type.
 */

#include <mpi.h>
#include "initialise/init_supers_from_binary.hpp"

template <typename T>
inline std::vector<T> nan_vector(const size_t size) {
  const auto nanValue = std::numeric_limits<T>::signaling_NaN();
  return std::vector<T>(size, nanValue);
}

/* sets sdIds for un-initialised superdrops' using an sdId's generator */
std::vector<Superdrop::IDType> InitSupersFromBinary::sdIds_for_uninitialised_superdrops(
    const size_t size) const {
  auto sdIdgen = Superdrop::IDType::Gen();

  auto sdIds = std::vector<Superdrop::IDType>(
      size, sdIdgen.set(std::numeric_limits<unsigned int>::signaling_NaN()));

  return sdIds;
}

/* adds data for un-initialised (and out of bounds) superdrops into initdata so that initial
conditions exist for maxnsupers number of superdrops in total */
InitSupersData InitSupersFromBinary::add_uninitialised_superdrops_data(
    InitSupersData &initdata) const {
  const auto size = maxnsupers - initdata.sdgbxindexes.size();

  const auto sdgbxindexes = std::vector<unsigned int>(size, LIMITVALUES::uintmax);  // out of bounds
  const auto coord3s = nan_vector<double>(size);
  const auto coord1s = nan_vector<double>(size);
  const auto coord2s = nan_vector<double>(size);
  const auto radii = nan_vector<double>(size);
  const auto msols = nan_vector<double>(size);
  const auto xis = nan_vector<uint64_t>(size);
  const auto sdIds = sdIds_for_uninitialised_superdrops(size);

  const auto nandata = InitSupersData{
      initdata.solutes, sdgbxindexes, coord3s, coord1s, coord2s, radii, msols, xis, sdIds};

  return initdata + nandata;
}

void InitSupersFromBinary::trim_nonlocal_superdrops(InitSupersData &initdata) const {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (gbxmaps.get_domain_decomposition().get_total_global_gridboxes() ==
        gbxmaps.get_domain_decomposition().get_total_local_gridboxes())
        return;

    unsigned int gridbox_index = 0;

    // Go through all superdrops and resets the values of the non-local ones
    for (size_t superdrop_index = 0; superdrop_index < initdata.sdgbxindexes.size();) {
        gridbox_index = initdata.sdgbxindexes[superdrop_index];
        if (my_rank != gbxmaps.get_domain_decomposition()
                              .get_gridbox_owner_process(gridbox_index)) {
            // resets superdrops which are in gridboxes not owned by this process
            initdata.sdgbxindexes[superdrop_index] = LIMITVALUES::uintmax;
            initdata.xis[superdrop_index] = std::numeric_limits<uint64_t>::signaling_NaN();;
            initdata.radii[superdrop_index] = std::numeric_limits<double>::signaling_NaN();;
            initdata.msols[superdrop_index] = std::numeric_limits<double>::signaling_NaN();;
            initdata.coord3s[superdrop_index] = std::numeric_limits<double>::signaling_NaN();;
            initdata.coord1s[superdrop_index] = std::numeric_limits<double>::signaling_NaN();;
            initdata.coord2s[superdrop_index] = std::numeric_limits<double>::signaling_NaN();;
        } else {
            // updates superdrop gridbox index from global to local
            initdata.sdgbxindexes[superdrop_index] = gbxmaps
                                                     .get_domain_decomposition()
                                                     .global_to_local_gridbox_index(gridbox_index);
        }
        superdrop_index++;
    }
}
