/*
 * ----- CLEO -----
 * File: cartesianmotion.hpp
 * Project: cartesiandomain
 * Created Date: Wednesday 8th November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Tuesday 19th December 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * Motion of a superdroplet using predictor-corrector
 * method to update a superdroplet's coordinates and
 * the sdgbxindex updated accordingly for a
 * cartesian domain with finite/periodi boundary
 * conditions
 */

#ifndef CARTESIANMOTION_HPP
#define CARTESIANMOTION_HPP

#include <functional>
#include <cassert>

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

#include "./cartesianboundaryconds.hpp"
#include "./cartesianmaps.hpp"
#include "superdrops/superdrop.hpp"
#include "superdrops/terminalvelocity.hpp"
#include "gridboxes/predcorrmotion.hpp"


KOKKOS_FUNCTION void
beyonddomain_backwards_coord3(const CartesianMaps &gbxmaps,
                              const unsigned int idx,
                              const unsigned int nghbr,
                              Superdrop &drop);

KOKKOS_FUNCTION void
beyonddomain_forwards_coord3(const CartesianMaps &gbxmaps,
                             const unsigned int idx,
                             const unsigned int nghbr,
                             Superdrop &drop);

KOKKOS_FUNCTION unsigned int
change_if_coord1nghbr(const CartesianMaps &gbxmaps,
                      unsigned int idx,
                      Superdrop &drop);

KOKKOS_FUNCTION unsigned int
change_if_coord2nghbr(const CartesianMaps &gbxmaps,
                      unsigned int idx,
                      Superdrop &drop);

struct CartesianCheckBounds
/* wrapper of operator for use of function
in PredCorrMotion's CheckBounds type */
{
  KOKKOS_INLINE_FUNCTION void
  operator()(const unsigned int idx,
             const Kokkos::pair<double, double> bounds,
             const double coord) const
  /* raise error if superdrop not either out of domain
  or within bounds (ie. lower_bound <= coord < upper_bound) */
  {
    const bool bad_gbxindex((idx != outofbounds_gbxindex()) &&
                            ((coord < bounds.first) | (coord >= bounds.second)));

    assert((!bad_gbxindex) && "SD not in previous gbx nor a neighbour."
                              " Try reducing the motion timestep to"
                              " satisfy CFL criteria, or use "
                              " 'update_ifoutside' to update sd_gbxindex");
  }
};

struct CartesianDomainBoundaryCoord3
{
  KOKKOS_FUNCTION unsigned int
  backwards(const unsigned int idx,
            const unsigned int nghbr,
            const CartesianMaps &gbxmaps,
            Superdrop &drop) const
  /* function to return gbxindex of neighbouring gridbox
  in backwards coord3 (z) direction and to update superdrop
  if its coord3 has exceeded the z lower domain boundary */
  {
    const auto incre = (unsigned int)1;                         // increment
    if (beyond_domainboundary(idx, incre, gbxmaps.get_ndim(0))) // drop was at lower z edge of domain (now moving below it)
    {
      beyonddomain_backwards_coord3(gbxmaps, idx, nghbr, drop);
    }

    return nghbr; // gbxindex of z backwards (down) neighbour
  };

  KOKKOS_FUNCTION unsigned int
  forwards(const unsigned int idx,
           const unsigned int nghbr,
           const CartesianMaps &gbxmaps,
           Superdrop &drop)
  /* function to return gbxindex of neighbouring gridbox in
  forwards coord3 (z) direction and to update superdrop coord3
  if superdrop has exceeded the z upper domain boundary */
  {
    const auto incre = (unsigned int)1;                                 // increment
    if (beyond_domainboundary(idx + incre, incre, gbxmaps.get_ndim(0))) // drop was upper z edge of domain (now moving above it)
    {
      beyonddomain_forwards_coord3(gbxmaps, idx, nghbr, drop);
    }

    return nghbr; // gbxindex of z forwards (up) neighbour
  };
};

struct CartesianFlagSdgbxindex
/* wrapper around flag_sdgbxindex function for use
as Flag type in ChangeIfNghbr struct */
{
  KOKKOS_FUNCTION int
  flag_sdgbxindex(const unsigned int idx,
                  const Kokkos::pair<double, double> bounds,
                  const double coord) const;

  KOKKOS_INLINE_FUNCTION int
  operator()(const unsigned int idx,
             const Kokkos::pair<double, double> bounds,
             const double coord) const;
  {
    return flag_sdgbxindex(idx, bounds, coord);
  }
};

template <VelocityFormula TV>
inline PredCorrMotion<CartesianMaps, TV, CartesianCheckBounds>
CartesianMotion(const unsigned int motionstep,
                const std::function<double(unsigned int)> int2time,
                const TV terminalv)
/* returned type satisfies motion concept for motion of a
superdroplet using a predictor-corrector method to update
a superdroplet's coordinates and then updating it's
sdgbxindex as appropriate for a cartesian domain */
{
  const auto flag = CartesianFlagSdgbxindex{};
  const auto change_if_nghbr = ChangeIfNghbr(flag);

  return PredCorrMotion<CartesianMaps, TV,
                        CartesianCheckBounds>(motionstep,
                                              int2time,
                                              terminalv,
                                              change_if_nghbr,
                                              CartesianCheckBounds{});
}

/* -----  ----- TODO: move functions below to .cpp file ----- ----- */
KOKKOS_FUNCTION void
beyonddomain_backwards_coord1(const CartesianMaps &gbxmaps,
                              const unsigned int idx,
                              const unsigned int nghbr,
                              Superdrop &drop);

KOKKOS_FUNCTION void
beyonddomain_forwards_coord1(const CartesianMaps &gbxmaps,
                             const unsigned int idx,
                             const unsigned int nghbr,
                             Superdrop &drop);

KOKKOS_FUNCTION void
beyonddomain_backwards_coord2(const CartesianMaps &gbxmaps,
                              const unsigned int idx,
                              const unsigned int nghbr,
                              Superdrop &drop);

KOKKOS_FUNCTION void
beyonddomain_forwards_coord2(const CartesianMaps &gbxmaps,
                             const unsigned int idx,
                             const unsigned int nghbr,
                             Superdrop &drop);

KOKKOS_FUNCTION int
CartesianFlagSdgbxindex::flag_sdgbxindex(const unsigned int idx,
                                         const Kokkos::pair<double, double> bounds,
                                         const double coord) const
/* returns flag to keep idx the same (flag = 0) or
update to forwards (flag = 1) or backwards (flag = 2)
neighbour. Flag = 0 if idx is out of domain value or
if coord lies within bounds = {lowerbound, upperbound}.
(Note: lower bound inclusive and upper bound exclusive,
ie. lowerbound <= coord < upperbound).
Flag = 1 if coord < lowerbound, indicating idx should
be updated to backwards neighbour.
Flag = 2 if coord >= upperbound, indicating idx should
be updated to forwards neighbour. */
{
  if (idx == outofbounds_gbxindex())
  {
    return 0; // maintian idx that is already out of domain
  }
  else if (coord < bounds.first) // lowerbound
  {
    return 1; // idx -> backwards_neighbour
  }
  else if (coord >= bounds.second) // upperbound
  {
    return 2; // idx -> forwards_neighbour
  }
  else
  {
    return 0; // maintain idx if coord within bounds
  }
}

KOKKOS_FUNCTION unsigned int
change_to_backwards_coord1nghbr(const unsigned int idx,
                                const CartesianMaps &gbxmaps,
                                Superdrop &drop)
/* function to return gbxindex of neighbouring gridbox
in backwards coord1 (x) direction and to update superdrop
coord1 if superdrop has exceeded the x back domain boundary */
{
  const auto nghbr = (unsigned int)gbxmaps.coord1backward(idx);

  const auto ndims(gbxmaps.get_ndims());
  const auto incre = (unsigned int)ndims(0);       // increment
  if (beyond_domainboundary(idx, incre, ndims(1))) // at lower x edge of domain
  {
    beyonddomain_backwards_coord1(gbxmaps, idx, nghbr, drop);
  }

  drop.set_sdgbxindex(nghbr);
  return nghbr; // gbxindex of x backwards (behind) neighbour
};

KOKKOS_FUNCTION unsigned int
change_to_forwards_coord1nghbr(const unsigned int idx,
                               const CartesianMaps &gbxmaps,
                               Superdrop &drop)
/* function to return gbxindex of neighbouring gridbox
in forwards coord1 (x) direction and to update superdrop
coord1 if superdrop has exceeded the x front domain boundary */
{
  const auto nghbr = (unsigned int)gbxmaps.coord1forward(idx);

  const auto ndims(gbxmaps.get_ndims());
  const auto incre = (unsigned int)ndims(0);               // increment
  if (beyond_domainboundary(idx + incre, incre, ndims(1))) // at lower x edge of domain
  {
    beyonddomain_forwards_coord1(gbxmaps, idx, nghbr, drop);
  }

  drop.set_sdgbxindex(nghbr);
  return nghbr; // gbxindex of x forwards (infront) neighbour
};

KOKKOS_FUNCTION unsigned int
change_to_backwards_coord2nghbr(const unsigned int idx,
                                const CartesianMaps &gbxmaps,
                                Superdrop &drop)
/* function to return gbxindex of neighbouring gridbox
in backwards coord2 (y) direction and to update superdrop
coord2 if superdrop has exceeded the y leftmost domain boundary */
{
  const auto nghbr = (unsigned int)gbxmaps.coord2backward(idx);

  const auto ndims(gbxmaps.get_ndims());
  const auto incre = (unsigned int)ndims(0) * ndims(1); // no. gridboxes in z direction * no. gridboxes in x direction
  if (beyond_domainboundary(idx, incre, ndims(2)))      // at lower y edge of domain
  {
    beyonddomain_backwards_coord2(gbxmaps, idx, nghbr, drop);
  }

  drop.set_sdgbxindex(nghbr);
  return nghbr; // gbxindex of y backwards (left) neighbour
};

KOKKOS_FUNCTION unsigned int
change_to_forwards_coord2nghbr(const unsigned int idx,
                               const CartesianMaps &gbxmaps,
                               Superdrop &drop)
/* function to return gbxindex of neighbouring gridbox
in forwards coord2 (y) direction and to update superdrop
coord2 if superdrop has exceeded the y rightmost domain boundary */
{
  const auto nghbr = (unsigned int)gbxmaps.coord2forward(idx);

  const auto ndims(gbxmaps.get_ndims());
  const auto incre = (unsigned int)ndims(0) * ndims(1);    // no. gridboxes in z direction * no. gridboxes in x direction
  if (beyond_domainboundary(idx + incre, incre, ndims(2))) // at upper y edge of domain
  {
    beyonddomain_forwards_coord2(gbxmaps, idx, nghbr, drop);
  }

  drop.set_sdgbxindex(nghbr);
  return nghbr; // gbxindex of y forwards (right) neighbour
};

KOKKOS_FUNCTION void
beyonddomain_backwards_coord3(const CartesianMaps &gbxmaps,
                              const unsigned int idx,
                              const unsigned int nghbr,
                              Superdrop &drop)
/* function updates superdrop that has crossed the
domain boundary in the backwards coord3 (z) direction
(i.e. superdrop has exceeded the z lower domain boundary) */
{
  const auto lim1 = double{gbxmaps.coord3bounds(nghbr).second}; // upper lim of backward neighbour
  const auto lim2 = double{gbxmaps.coord3bounds(idx).first};    // lower lim of current gbx
  drop.set_coord3(boundarycond_coord3(drop.get_coord3(), lim1, lim2));
}

KOKKOS_FUNCTION void
beyonddomain_forwards_coord3(const CartesianMaps &gbxmaps,
                             const unsigned int idx,
                             const unsigned int nghbr,
                             Superdrop &drop)
/* function updates superdrop that has crossed the
domain boundary in the forwards coord3 (z) direction
(i.e. superdrop has exceeded the z upper domain boundary) */
{
  const auto lim1 = double{gbxmaps.coord3bounds(nghbr).first}; // lower lim of forward neighbour
  const auto lim2 = double{gbxmaps.coord3bounds(idx).second};  // upper lim of current gbx
  drop.set_coord3(boundarycond_coord3(drop.get_coord3(), lim1, lim2));
}

KOKKOS_FUNCTION void
beyonddomain_backwards_coord1(const CartesianMaps &gbxmaps,
                              const unsigned int idx,
                              const unsigned int nghbr,
                              Superdrop &drop)
/* function updates superdrop that has crossed the
domain boundary in the backwards coord1 (x) direction
(i.e. superdrop has exceeded the x back domain boundary) */
{
  const auto lim1 = double{gbxmaps.coord1bounds(nghbr).second}; // upper lim of backward neigghbour
  const auto lim2 = double{gbxmaps.coord1bounds(idx).first};    // lower lim of current gbx
  drop.set_coord1(boundarycond_coord1(drop.get_coord1(), lim1, lim2));
}

KOKKOS_FUNCTION void
beyonddomain_forwards_coord1(const CartesianMaps &gbxmaps,
                             const unsigned int idx,
                             const unsigned int nghbr,
                             Superdrop &drop)
/* function updates superdrop that has crossed the
domain boundary in the forwards coord1 (x) direction
(i.e. superdrop has exceeded the x front domain boundary) */
{
  const auto lim1 = double{gbxmaps.coord1bounds(nghbr).first}; // lower lim of forward nghbour
  const auto lim2 = double{gbxmaps.coord1bounds(idx).second};  // upper lim of gbx
  drop.set_coord1(boundarycond_coord1(drop.get_coord1(), lim1, lim2));
}

KOKKOS_FUNCTION void
beyonddomain_backwards_coord2(const CartesianMaps &gbxmaps,
                              const unsigned int idx,
                              const unsigned int nghbr,
                              Superdrop &drop)
/* function updates superdrop that has crossed the
domain boundary in the backwards coord2 (y) direction
(i.e. superdrop has exceeded the y leftmost domain boundary) */
{
  const auto lim1 = double{gbxmaps.coord2bounds(nghbr).second}; // upper lim of backward nghbour
  const auto lim2 = double{gbxmaps.coord2bounds(idx).first};    // lower lim of gbx
  drop.set_coord2(boundarycond_coord2(drop.get_coord2(), lim1, lim2));
}

KOKKOS_FUNCTION void
beyonddomain_forwards_coord2(const CartesianMaps &gbxmaps,
                             const unsigned int idx,
                             const unsigned int nghbr,
                             Superdrop &drop)
/* function updates superdrop that has crossed the
domain boundary in the forwards coord2 (y) direction
(i.e. superdrop has exceeded the y rightmost domain boundary) */
{
  const auto lim1 = double{gbxmaps.coord2bounds(nghbr).first}; // lower lim of forward nghbour
  const auto lim2 = double{gbxmaps.coord2bounds(idx).second};  // upper lim of gbx
  drop.set_coord2(boundarycond_coord2(drop.get_coord2(), lim1, lim2));
}

#endif // CARTESIANMOTION_HPP