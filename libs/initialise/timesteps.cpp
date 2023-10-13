/*
 * ----- CLEO -----
 * File: timesteps.cpp
 * Project: initialise
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Friday 13th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * functionality for handling model timesteps and
 * their conversions to/from real times
 */


#include "./timesteps.hpp"

Timesteps::Timesteps(const Config &config)
    /* (dimensionless) double's that are timesteps in config struct
    are converted into integer values of model timesteps using
    model_step and secd template functions created using std::chrono library.
    Throw error if after convertion into model timestep, any
    timestep = 0. Substeps for sdmprocess must be larger than steps! */
    : condstep(realtime2step(config.CONDTSTEP)),
      collstep(realtime2step(config.COLLTSTEP)),
      motionstep(realtime2step(config.MOTIONTSTEP)),
      couplstep(realtime2step(config.COUPLTSTEP)),
      obsstep(realtime2step(config.OBSTSTEP)),
      t_end(realtime2step(config.T_END))
{
  if ((condstep == 0) | (collstep == 0) | (motionstep == 0) |
      (couplstep == 0) | (obsstep == 0) | (t_end == 0))
  {
    const std::string err("A model step = 0, possibly due to bad"
                          "conversion of a real timestep [s]. Consider"
                          " increasing X in std::ratio<1, X> used for"
                          " definition of model_step");
    throw std::invalid_argument(err);
  }

  const unsigned int maxsubstep(std::max(condstep, collstep));
  const unsigned int minstep = std::min(std::min(couplstep, obsstep),
                                        motionstep);
  if (minstep < maxsubstep)
  {
    const std::string err("invalid SDM substepping: an SDM substep"
                          " is larger than the smallest step"
                          " (coupling, observation or motion step)");
    throw std::invalid_argument(err);
  }

  if (std::min(couplstep, obsstep) < motionstep)
  {
    const std::string err("Warning: coupling / observation step is"
                          "smaller than the sdmmotion step"
                          " - are you really sure you want this?");
    throw std::invalid_argument(err);
  }
}