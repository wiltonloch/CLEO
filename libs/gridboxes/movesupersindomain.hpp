/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 * ----- CLEO -----
 * File: movesupersindomain.hpp
 * Project: gridboxes
 * Created Date: Friday 13th October 2023
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
 * Functionality related to moving superdroplets
 * (both updating their spatial coordinates and
 * moving them between gridboxes)
 */

#ifndef LIBS_GRIDBOXES_MOVESUPERSINDOMAIN_HPP_
#define LIBS_GRIDBOXES_MOVESUPERSINDOMAIN_HPP_

#include <Kokkos_Core.hpp>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <numeric>
#include "mpi.h"

#include "../cleoconstants.hpp"
#include "../kokkosaliases.hpp"
#include "./gridbox.hpp"
#include "./gridboxmaps.hpp"
#include "./sortsupers.hpp"
#include "superdrops/motion.hpp"
#include "superdrops/superdrop.hpp"

/* struct for functionality to move superdroplets throughtout
the domain by updating their spatial coordinates (according to
some type of Motion) and then moving them between gridboxes
after updating their gridbox indexes concordantly */
template <GridboxMaps GbxMaps, Motion<GbxMaps> M, typename BoundaryConditions>
struct MoveSupersInDomain {
  M motion;
  BoundaryConditions apply_domain_boundary_conditions;

  /* enact steps (1) and (2) movement of superdroplets for 1 gridbox:
  (1) update their spatial coords according to type of motion. (device)
  (1b) optional detect precipitation (device)
  (2) update their sdgbxindex accordingly (device).
  Kokkos::parallel_for([...]) is equivalent to:
  for (size_t kk(0); kk < supers.extent(0); ++kk) {[...]}
  when in serial */
  KOKKOS_INLINE_FUNCTION
  void move_supers_in_gbx(const TeamMember &team_member, const unsigned int gbxindex,
                          const GbxMaps &gbxmaps, const State &state,
                          const subviewd_supers supers) const {
    const size_t nsupers(supers.extent(0));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nsupers), [&, this](const size_t kk) {
      /* step (1) */
      motion.superdrop_coords(gbxindex, gbxmaps, state, supers(kk));

      /* optional step (1b) */
      // gbx.detectors -> detect_precipitation(area, drop); // TODO(CB) detectors

      /* step (2) */
      motion.superdrop_gbx(gbxindex, gbxmaps, supers(kk));
    });
  }

  /* enact steps (1) and (2) movement of superdroplets
  throughout domain (i.e. for all gridboxes):
  (1) update their spatial coords according to type of motion. (device)
  (1b) optional detect precipitation (device)
  (2) update their sdgbxindex accordingly (device).
  Kokkos::parallel_for([...]) is equivalent to:
  for (size_t ii(0); ii < ngbxs; ++ii) {[...]}
  when in serial */
  void move_supers_in_gridboxes(const GbxMaps &gbxmaps, const viewd_gbx d_gbxs) const {
    const size_t ngbxs(d_gbxs.extent(0));

    Kokkos::parallel_for(
        "move_supers_in_gridboxes", TeamPolicy(ngbxs, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const TeamMember &team_member) {
          const int ii = team_member.league_rank();

          auto &gbx(d_gbxs(ii));
          move_supers_in_gbx(team_member, gbx.get_gbxindex(), gbxmaps, gbx.state,
                             gbx.supersingbx());
        });
  }

  /* (re)sorting supers based on their gbxindexes and then updating the span for each gridbox
  accordingly.
  Kokkos::parallel_for([...]) (on host) is equivalent to:
  for (size_t ii(0); ii < ngbxs; ++ii){[...]}
  when in serial
  _Note:_ totsupers is view of all superdrops (both in and out of bounds of domain).
  */
  void move_supers_between_gridboxes(const viewd_gbx d_gbxs, viewd_supers totsupers) const {
    sort_supers(totsupers);

    const size_t ngbxs(d_gbxs.extent(0));

    int comm_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    std::vector<int> per_process_send_superdrops(comm_size, 0);
    std::vector<int> per_process_recv_superdrops(comm_size, 0);
    std::vector<int> uint_send_displacements(comm_size, 0);
    std::vector<int> uint_recv_displacements(comm_size, 0);
    std::vector<int> double_send_displacements(comm_size, 0);
    std::vector<int> double_recv_displacements(comm_size, 0);
    std::vector<int> double_send_counts(comm_size, 0);
    std::vector<int> double_recv_counts(comm_size, 0);
    std::vector<std::vector<int>> superdrops_indices_per_process(comm_size);

    size_t total_superdrops_to_send      = 0;
    size_t total_superdrops_to_recv      = 0;
    size_t local_superdrops              = 0;
    size_t superdrop_index               = totsupers.extent(0) - 1;
    Superdrop & drop                     = totsupers(superdrop_index);
    per_process_send_superdrops[my_rank] = 0;

    // Go through superdrops from back to front and find how many should be sent and their indices
    while(drop.get_sdgbxindex() > ngbxs) {
        if(drop.get_sdgbxindex() < comm_size * 2 * ngbxs) {
            per_process_send_superdrops  [ (drop.get_sdgbxindex() - ngbxs) / ngbxs]++;
            superdrops_indices_per_process[ (drop.get_sdgbxindex() - ngbxs) / ngbxs].push_back(superdrop_index);
            total_superdrops_to_send++;
        }
        drop = totsupers(--superdrop_index);
    }
    local_superdrops = superdrop_index + 1;

    // Knowing how many superdroplets will be sent, allocate buffers to serialize their data
    std::vector<double> superdrops_double_send_data(total_superdrops_to_send * 5);
    std::vector<unsigned int> superdrops_uint_send_data(total_superdrops_to_send);
    std::vector<uint64_t> superdrops_uint64_send_data(total_superdrops_to_send);

    // Serialize the data for all superdroplets into the exchange arrays
    for(int process_index = 0; process_index < comm_size; process_index++)
        for(int superdrop = 0; superdrop < per_process_send_superdrops[process_index]; superdrop++){
            superdrop_index = superdrops_indices_per_process[process_index][superdrop];
            totsupers[superdrop_index].serialize_uint_components(superdrops_uint_send_data.begin() + superdrop);
            totsupers[superdrop_index].serialize_uint64_components(superdrops_uint64_send_data.begin() + superdrop);
            totsupers[superdrop_index].serialize_double_components(superdrops_double_send_data.begin() + superdrop * 5);
        }

    // Share how many superdrops each process will send and receive to/from the others
    MPI_Alltoall(per_process_send_superdrops.data(), 1, MPI_INT, per_process_recv_superdrops.data(), 1, MPI_INT, MPI_COMM_WORLD);
    total_superdrops_to_recv = std::accumulate(per_process_recv_superdrops.begin(), per_process_recv_superdrops.end(), 0);

    // Knowing how many superdroplets will be received, allocate buffers to receive data
    std::vector<double> superdrops_double_recv_data(total_superdrops_to_recv * 5);
    std::vector<unsigned int> superdrops_uint_recv_data(total_superdrops_to_recv);
    std::vector<uint64_t> superdrops_uint64_recv_data(total_superdrops_to_recv);

    // Calculate the send and receive counts and displacements for each of the target processes
    for (int i = 0; i < comm_size; i++) {
        double_send_counts[i] = per_process_send_superdrops[i] * 5;
        double_recv_counts[i] = per_process_recv_superdrops[i] * 5;

        if(i > 0) {
            uint_send_displacements[i] = uint_send_displacements[i - 1] + per_process_send_superdrops[i - 1];
            uint_recv_displacements[i] = uint_recv_displacements[i - 1] + per_process_recv_superdrops[i - 1];
            double_send_displacements[i] = double_send_displacements[i - 1] + double_send_counts[i - 1];
            double_recv_displacements[i] = double_recv_displacements[i - 1] + double_recv_counts[i - 1];
        }
    }

    // Exchange superdrops uint data
    MPI_Alltoallv(superdrops_uint_send_data.data(), per_process_send_superdrops.data(),
                  uint_send_displacements.data(), MPI_INT,
                  superdrops_uint_recv_data.data(), per_process_recv_superdrops.data(),
                  uint_recv_displacements.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Exchange superdrops uint64 data
    MPI_Alltoallv(superdrops_uint64_send_data.data(), per_process_send_superdrops.data(),
                  uint_send_displacements.data(), MPI_INT,
                  superdrops_uint64_recv_data.data(), per_process_recv_superdrops.data(),
                  uint_recv_displacements.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Exchange superdrops double data
    MPI_Alltoallv(superdrops_double_send_data.data(), double_send_counts.data(),
                  double_send_displacements.data(), MPI_INT,
                  superdrops_double_recv_data.data(), double_recv_counts.data(),
                  double_recv_displacements.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // if(my_rank == 0)
    //     std::cout << totsupers.extent(0) << std::endl;

    if (local_superdrops + total_superdrops_to_recv > totsupers.extent(0)){
        std::cout << "MAXIMUM NUMBER OF LOCAL SUPERDROPLETS EXCEEDED" << std::endl;
        return;
    }

    // std::cout << my_rank << " " << total_superdrops_to_send << " " << total_superdrops_to_recv << std::endl;
    //
    // MPI_Barrier(MPI_COMM_WORLD);

    // // Converts global gridbox index to local
    // for (auto &i : superdrops_uint_recv_data){
    //     i -= ngbxs;
    // }
    //
    // for(unsigned int i = local_superdrops; i < local_superdrops + total_superdrops_to_recv; i++){
    //     totsupers[i].deserialize_components(superdrops_uint_recv_data.begin()   + i,
    //                                         superdrops_uint64_recv_data.begin() + i,
    //                                         superdrops_double_recv_data.begin() + i * 5);
    //     // if(my_rank == 1)
    //     //     std::cout << "Setting " << i << " from " << local_superdrops + total_superdrops_to_recv << std::endl;
    //     //     std::cout << "Local superdrop gbx " << totsupers[i].get_sdgbxindex() << std::endl;
    // }
    //
    // for(unsigned int i = local_superdrops + total_superdrops_to_recv; i < totsupers.extent(0); i++) {
    //     totsupers[i].set_sdgbxindex(LIMITVALUES::uintmax);
    // }

    sort_supers(totsupers);

    Kokkos::parallel_for(
        "move_supers_between_gridboxes", TeamPolicy(ngbxs, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const TeamMember &team_member) {
          const int ii = team_member.league_rank();

          auto &gbx(d_gbxs(ii));
          gbx.supersingbx.set_refs(team_member);
    });

    // /* optional (expensive!) test to raise error if
    // superdrops' gbxindex doesn't match gridbox's gbxindex */
    // for (size_t ii(0); ii < ngbxs; ++ii)
    // {
    //   d_gbxs(ii).supersingbx.iscorrect();
    // }
  }

  /* enact movement of superdroplets throughout domain in three stages:
  (1) update their spatial coords according to type of motion. (device)
  (1b) optional detect precipitation (device)
  (2) update their sdgbxindex accordingly (device)
  (3) move superdroplets between gridboxes (host)
  (4) (optional) apply domain boundary conditions (host and device)
  _Note:_ totsupers is view of all superdrops (both in and out of bounds of domain).
  // TODO(all) use tasking to convert all 3 team policy
  // loops from first two function calls into 1 loop?
  */
  void move_superdrops_in_domain(const unsigned int t_sdm, const GbxMaps &gbxmaps, viewd_gbx d_gbxs,
                                 viewd_supers totsupers) const {
    /* steps (1 - 2) */
    move_supers_in_gridboxes(gbxmaps, d_gbxs);

    /* step (3) */
    move_supers_between_gridboxes(d_gbxs, totsupers);

    /* step (4) */
    apply_domain_boundary_conditions(gbxmaps, d_gbxs, totsupers);
  }

  MoveSupersInDomain(const M mtn, const BoundaryConditions boundary_conditions)
      : motion(mtn), apply_domain_boundary_conditions(boundary_conditions) {}

  /* extra constructor useful to help when compiler cannot deduce type of GBxMaps */
  MoveSupersInDomain(const GbxMaps &gbxmaps, const M mtn,
                     const BoundaryConditions boundary_conditions)
      : MoveSupersInDomain(mtn, boundary_conditions) {}

  /* returns time when superdroplet motion is
  next due to occur given current time, t_sdm */
  KOKKOS_INLINE_FUNCTION
  unsigned int next_step(const unsigned int t_sdm) const { return motion.next_step(t_sdm); }

  /*
   * if current time, t_sdm, is time when superdrop motion should occur, enact movement of
   * superdroplets throughout domain.
   *
   * @param totsupers View of all superdrops (both in and out of bounds of domain).
   *
   */
  void run_step(const unsigned int t_sdm, const GbxMaps &gbxmaps, viewd_gbx d_gbxs,
                viewd_supers totsupers) const {
    if (motion.on_step(t_sdm)) {
      move_superdrops_in_domain(t_sdm, gbxmaps, d_gbxs, totsupers);
    }
  }
};

#endif  // LIBS_GRIDBOXES_MOVESUPERSINDOMAIN_HPP_
