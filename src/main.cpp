// Author: Clara Bayley
// File: main.cpp
/* This file runs the entire superdrop model (SDM)
coupled with a CVODE ode solver for the thermodynamics
(p, temp, qv and qc) over time */

// after make/compiling, execute for example via:
// ./src/coupledCVODECLEO "../src/config/config.txt" "../libs/claras_SDconstants.hpp"

#include "main_supplement.hpp"

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    throw std::invalid_argument("config and/or constants files not specified");
    return -1;
  }

  /* object containing input parameters from configuration file */
  const std::string configfilepath = argv[1];    // path to configuration (.txt file)
  const std::string constantsfilepath = argv[2]; // path to constants (.hpp file)
  const Config config(configfilepath, constantsfilepath);

  /* object for time-stepping parameters of coupled model */
  const ModelTimesteps mdlsteps(config.CONDTSTEP, config.COLLTSTEP,
                                config.MOTIONTSTEP, config.COUPLTSTEP,
                                config.T_END);

  /* create map from gridbox index to its coordinate boundaries */
  const Maps4GridBoxes gbxmaps(config.SDnspace, config.grid_filename);

  /* create superdroplet model (SDM) process from combination of chosen SDM processes */
  const auto sdmprocess = create_sdmprocess(config, mdlsteps);

  /* create observer from combination of chosen observers */
  // std::ofstream thermo_datafile, superdrops_datafile;
  // const auto observer = create_observer(config, thermo_datafile, superdrops_datafile);
  const unsigned int ngridboxes = gbxmaps.gbxidxs.size();
  FSStore fsstore(config.zarrbasedir);

  //SomeZarrStorage zarrstores()

  ContiguousRaggedSDStorage sdzarr(fsstore, superdropattributes_to_observe(),
                                          config.maxcsize);
  ThermoStateStorage thermozarr(fsstore, config.maxcsize, ngridboxes);
  CoordinateStorage<double> timezarr(fsstore, config.maxcsize,
                                "time", "<f8", "s", dlc::TIME0);
  CoordinateStorage<unsigned int> gbxzarr(fsstore, config.maxcsize,
                                     "gbxindex", "<u4", " ", 1);
  TwoDStorage<size_t> nsuperszarr(fsstore, config.maxcsize,
                                  "nsupers", "<u8", " ", 1, ngridboxes);
  SDMomentsStorage sdmoments(fsstore, config.maxcsize, ngridboxes);

  const auto observer = create_observer(config, sdzarr, thermozarr,
                                        timezarr, gbxzarr, nsuperszarr,
                                        sdmoments);

  const MoveSuperdropsInDomain sdmmotion(create_sdmotion(mdlsteps.motionstep));

  /* RUN SDM MODEL COUPLED TO CVODE ODE SOLVER */
  run_cvodeSDM_coupledmodel(config, gbxmaps, sdmmotion, sdmprocess,
                            observer, mdlsteps.t_end, mdlsteps.couplstep);

  return 0;
}