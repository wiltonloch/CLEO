/*
 * ----- CLEO -----
 * File: cvodecomms.cpp
 * Project: coupldyn_cvode
 * Created Date: Friday 27th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Sunday 29th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * send and receive dynamics functions for
 * SDM when coupled to the CVODE ODE solver
 */

#include "./cvodecomms.hpp"

void CvodeComms::receive_dynamics(const CvodeDynamics &cvode,
                                 const viewh_gbx h_gbxs) const
/* update Gridboxes' states using
information received from CVODE dynanmics
solver for  press, temp, qvap and qcond */
{
  const size_t ngbxs(h_gbxs.extent(0));
  for (size_t ii(0); ii < ngbxs; ++ii)
  {
    State &state(h_gbxs(ii).state);
    const auto cvodestate(cvode.get_current_state(ii)); // vector of states' [p, t, qv, qc]

    state.press = cvodestate.at(0);
    state.temp = cvodestate.at(1);
    state.qvap = cvodestate.at(2);
    state.qcond = cvodestate.at(3);
  }
}

void CvodeComms::send_dynamics(const viewh_constgbx h_gbxs,
                            CvodeDynamics &cvode) const
/* send information from Gridboxes' states
to CVODE dynanmics solver for  temp, qvap
and qcond (excludes press) */
{
  std::vector<double> delta_y;
  bool is_delta_y(false);

  const size_t ngbxs(h_gbxs.extent(0));
  for (size_t ii(0); ii<ngbxs; ++ii)
  {
    const auto delta(state_change(cvode, h_gbxs, ii)); // ii'th [press, temp, qvap, qcond] change

    delta_y.push_back(delta.at(0));
    delta_y.push_back(delta.at(1));
    delta_y.push_back(delta.at(2));
    delta_y.push_back(delta.at(3));

    is_delta_y = is_state_change(delta, is_delta_y);
  }

  if (is_delta_y)
  {
    cvode.reinitialise(cvode.get_time(), delta_y);
  }
}

std::array<double, 4>
CvodeComms::state_change(CvodeDynamics &cvode,
                         const viewh_constgbx h_gbxs,
                         const size_t ii) const
/* get change in state since
previous time step to current one */
{
  const State state(h_gbxs(ii).state);
  const auto prevstate(cvode.get_previous_state(ii)); // [press, temp, qvap, qcond]

  std::array<double, 4> delta;
  delta.at(0) = 0.0; // assume no change to press
  delta.at(1) = state.temp - prevstate.at(1);
  delta.at(2) = state.qvap - prevstate.at(2);
  delta.at(3) = state.qcond - prevstate.at(3);

  return delta;
}

bool CvodeComms::is_state_change(const std::array<double, 4> &delta,
                                 bool is_delta_y) const
/* change is_delta_y = false to is_delta_y = true
if delta contains non-zero elements */
{
  if (is_delta_y == false)
  {
    const std::array<double, 4> nodelta = {0.0, 0.0, 0.0, 0.0};
    if (delta != nodelta)
    {
      is_delta_y = true;
    }
  }

  return is_delta_y;
}