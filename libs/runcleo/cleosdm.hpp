/*
 * ----- CLEO -----
 * File: cleosdm.hpp
 * Project: runcleo
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
 * struct wrapping the core ingredients of the Super-droplet Model
 * (SDM) part of CLEO to enact on super-droplets and gridboxes
 */

#ifndef CLEOSDM_HPP
#define CLEOSDM_HPP

#include <string>
#include <stdexcept>

#include "./coupleddynamics.hpp"
#include "initialise/config.hpp"
#include "initialise/timesteps.hpp"
#include "sdmdomain/gridbox.hpp"
#include "sdmdomain/gridboxmaps.hpp"
#include "sdmdomain/movesupersindomain.hpp"
#include "superdrops/superdrop.hpp"
#include "superdrops/microphysicsprocess.hpp"
#include "observers/observers.hpp"

Gridboxes generate_gridboxes()
{
  return Gridboxes{};
}

Superdrops generate_superdrops()
{
  return Superdrops{};
}

struct CLEOSDM
{
private:
  unsigned int next_sdmstep(const unsigned int t_sdm,
                            const unsigned int stepsize) const;
  /* given current timestep, t_sdm, work out which event
  (motion or one complete step) is next to occur and return
  the time of the sooner event, (ie. next t_move or t_mdl) */

  void superdrops_movement(const unsigned int t_mdl,
                           Gridboxes &gbxs,
                           Superdrops &supers) const;
  /* move superdroplets (including movement between gridboxes)
  according to movesupers struct */

  void sdm_microphysics(const unsigned int t_sdm,
                        const unsigned int t_next,
                        Gridboxes &gbxs) const;
  /* enact SDM microphysics for each gridbox
  (using sub-timestepping routine) */

public:
  GridboxMaps gbxmaps;           // maps from gridbox indexes to domain coordinates
  MicrophysicsProcess microphys; // microphysical process
  MoveSupersInDomain movesupers; // super-droplets' motion in domain
  Observer obs;                  // observer
  unsigned int couplstep;

  CLEOSDM(const Config &config, const Timesteps &tsteps,
          const unsigned int coupldynstep)
      : gbxmaps(config), microphys(),
        movesupers(config, tsteps), obs(),
        couplstep(tsteps.get_couplstep())
  {
    if (couplstep != coupldynstep)
    {
      const std::string err("coupling timestep of dyanmics "
                            "solver and CLEO SDM are not equal");
      throw std::invalid_argument(err);
    }
  }

  unsigned int get_couplstep() const { return couplstep; }

  void prepare_to_timestep(const Gridboxes &gbxs,
                           const Superdrops &supers) const;
  /* prepare CLEO SDM for timestepping */

  void receive_dynamics(const CoupledDynamics &coupldyn,
                        Gridboxes &gbxs) const;
  /* update Gridboxes' states using information
  received from coupldyn */

  void send_dynamics(const CoupledDynamics &coupldyn,
                     Gridboxes &gbxs) const;
  /* send information from Gridboxes' states to coupldyn */

  void run_step(const unsigned int t_mdl,
                const unsigned int stepsize) const;
  /* run CLEO SDM from time t_mdl to t_mdl + stepsize with
  sub-timestepping routine for super-droplets' movement
  and microphysics */
};

#endif // CLEOSDM_HPP