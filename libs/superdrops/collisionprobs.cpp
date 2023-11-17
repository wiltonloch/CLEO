/*
 * ----- CLEO -----
 * File: collisionprobs.cpp
 * Project: superdrops
 * Created Date: Thursday 9th November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Thursday 16th November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * functionality to calculate the probability of some
 * kind of collision event between two (real) droplets
 * e.g. the probability of collision-coalescence
 * using the Golovin or Long Kernel. Probability
 * calculations are contained in structures
 * that satisfy the requirements of the
 * PairProbability concept (see collisions.hpp)
*/

#include "./collisionprobs.hpp"