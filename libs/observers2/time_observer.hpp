/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: time_observer.hpp
 * Project: observers2
 * Created Date: Wednesday 24th January 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 28th March 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Observer to output time at the start of each observation timestep to an array in a dataset
 */

#ifndef LIBS_OBSERVERS2_TIME_OBSERVER_HPP_
#define LIBS_OBSERVERS2_TIME_OBSERVER_HPP_

#include <Kokkos_Core.hpp>
#include <concepts>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>

#include "../kokkosaliases.hpp"
#include "./observers.hpp"
#include "gridboxes/gridbox.hpp"
#include "zarr2/dataset.hpp"
#include "zarr2/xarray_zarr_array.hpp"

/* template class for observing time and writing to array as a coordinate of an xarray dataset */
template <typename Store>
class DoTimeObs {
 private:
  Dataset<Store> &dataset;                                   ///< dataset to write time data to
  std::shared_ptr<XarrayZarrArray<Store, float>> xzarr_ptr;  ///< pointer to time array in dataset
  std::function<double(unsigned int)>
      step2dimlesstime;  // function to convert timesteps to real time [assumed seconds]

  // increment size of time dimension in dataset and write out time data to array in the dataset.
  // Note conversion of time from double (8 bytes) to single precision (4 bytes float) in output
  void at_start_step(const unsigned int t_mdl) const {
    const auto ntimes = size_t{dataset.get_dimension("time") + 1};
    const auto timedim = std::pair<std::string, size_t>({"time", ntimes});
    dataset.set_dimension(timedim);

    const auto time = static_cast<float>(step2dimlesstime(t_mdl));
    dataset.write_to_array(xzarr_ptr, time);
  }

 public:
  DoTimeObs(Dataset<Store> &dataset, const size_t maxchunk,
            const std::function<double(unsigned int)> step2dimlesstime)
      : dataset(dataset),
        xzarr_ptr(
            std::make_shared<XarrayZarrArray<Store, float>>(dataset.template create_array<float>(
                "time", "s", "<f4", dlc::TIME0, {maxchunk}, {"time"}))),
        step2dimlesstime(step2dimlesstime) {}

  ~DoTimeObs() { dataset.write_arrayshape(xzarr_ptr); }

  void before_timestepping(const viewd_constgbx d_gbxs) const {
    std::cout << "observer includes time observer\n";
  }

  void after_timestepping() const {}

  void at_start_step(const unsigned int t_mdl, const viewd_constgbx d_gbxs,
                     const viewd_constsupers totsupers) const {
    at_start_step(t_mdl);
  }
};

/* constructs observer which writes time variable out to a 1-D array with a constant timestep
'interval' using an instance of the ConstTstepObserver class */
template <typename Store>
inline Observer auto TimeObserver(const unsigned int interval, Dataset<Store> &dataset,
                                  const int maxchunk,
                                  const std::function<double(unsigned int)> step2dimlesstime) {
  dataset.add_dimension({"time", 0});
  return ConstTstepObserver(interval, DoTimeObs(dataset, maxchunk, step2dimlesstime));
}

#endif  // LIBS_OBSERVERS2_TIME_OBSERVER_HPP_
