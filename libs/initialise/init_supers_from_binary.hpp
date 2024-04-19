/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: init_supers_from_binary.hpp
 * Project: initialise
 * Created Date: Friday 19th April 2024
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

#ifndef LIBS_INITIALISE_INIT_SUPERS_FROM_BINARY_HPP_
#define LIBS_INITIALISE_INIT_SUPERS_FROM_BINARY_HPP_

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "./initconds.hpp"
#include "./optional_config_params.hpp"
#include "./readbinary.hpp"
#include "superdrops/superdrop_attrs.hpp"

/* struct containing functions which return data
for the initial conditions needed to create
superdroplets e.g. via the CreateSupers struct */
struct InitSupersFromBinary {
 private:
  size_t maxnsupers;  /**< total number of super-droplets (in kokkos view on device initially) */
  size_t initnsupers; /**< initial no. of super-droplets to initialise */
  std::filesystem::path initsupers_filename; /**< filename for super-droplets' initial conditons */
  unsigned int nspacedims; /**< number of spatial dimensions to model (0-D, 1-D, 2-D of 3-D) */

 public:
  InitAllSupersFromBinary(const size_t maxnsupers,
                          const OptionalConfigParams::InitAllSupersFromBinaryParams &config)
      : maxnsupers(maxnsupers),
        initnsupers(config.initnsupers),
        initsupers_filename(config.initsupers_filename),
        nspacedims(config.nspacedims) {
    assert((maxnsupers >= initnsupers) &&
           "cannot initialise more than the total number of super-droplets")
  }

  auto get_maxnsupers() const { return maxnsupers; }

  auto get_nspacedims() const { return nspacedims; }

  /* return InitSupersData created by reading some data from a binary file and
  filling the rest with invalid super-droplets */
  void fetch_data(InitSupersData &initdata) const {
    fetch_superdrops_from_file();
    fetch_invalid_superdrops_data();
  }
};

#endif  //  LIBS_INITIALISE_INIT_SUPERS_FROM_BINARY_HPP_
