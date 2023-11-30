/*
 * ----- CLEO -----
 * File: main.cpp
 * Project: roughpaper
 * Created Date: Wednesday 1st November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Friday 17th November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * rough paper for checking small things
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <limits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NestedSort.hpp>

struct Superdrop
{
private:
  unsigned int sdgbxindex;
public:

KOKKOS_INLINE_FUNCTION
void set_sdgbxindex(const unsigned int ii){sdgbxindex = ii;}

KOKKOS_INLINE_FUNCTION
    unsigned int get_sdgbxindex() const
{
  return sdgbxindex;
}
};

using viewd_supers = Kokkos::View<Superdrop *>;
using viewd_constsupers = Kokkos::View<const Superdrop *>; // view in device memory of const superdroplets
using ExecSpace = Kokkos::DefaultExecutionSpace;

viewd_supers init_supers(const size_t nsupers,
                         const size_t ngbxs)
{
    viewd_supers supers("supers", nsupers);
    auto h_supers = Kokkos::create_mirror_view(supers);
    for (size_t kk(0); kk < nsupers; ++kk)
    {
      const unsigned int ii(1);
      h_supers(kk) = Superdrop();
      h_supers(kk).set_sdgbxindex(ii);
      std::cout << "ii: " << h_supers(kk).get_sdgbxindex() << "\n";
    }
    Kokkos::deep_copy(supers, h_supers);

  return supers;
}

std::array<double, 3>
calc_massmoments(const viewd_constsupers d_supers)
/* calculated 0th, 1st and 2nd moment of the
(real) droplet mass (I.e. 0th, 3rd and 6th moment
of the droplet radius distribution).
Kokkos::parallel_for([...]) is equivalent in serial to:
for (size_t kk(0); kk < d_supers.extent(0); ++kk){[...]} */
{
  std::array<double, 3> moms({0.0, 0.0, 0.0}); // {0th, 1st, 2nd} mass moments

  const size_t nsupers(d_supers.extent(0));
  Kokkos::parallel_reduce(
      "massmoments_to_storage",
      Kokkos::RangePolicy<ExecSpace>(0, nsupers),
      KOKKOS_LAMBDA(const size_t kk, double &m0, double &m1, double &m2) {
        m0 += d_supers(kk).get_sdgbxindex();
        m1 += d_supers(kk).get_sdgbxindex() + 1;
        m2 += 0.0;
      },
      moms.at(0), moms.at(1), moms.at(2)); // {0th, 1st, 2nd} mass moments

  return moms;
}

int main(int argc, char *argv[])
{
  const size_t nsupers(10);
  const size_t ngbxs(1);

  Kokkos::initialize(argc, argv);
  {
    auto supers = init_supers(nsupers, ngbxs);

    for (size_t ii(0); ii < ngbxs; ++ii)
    {
      double mom0(0);
      double mom1(0);
      auto h_supers = Kokkos::create_mirror_view(supers);
      Kokkos::deep_copy(h_supers, supers);
      for (size_t kk(0); kk < nsupers; ++kk)
      {
        mom0 += h_supers(kk).get_sdgbxindex();
        mom1 += h_supers(kk).get_sdgbxindex() + 5;
      }
      std::cout << "ii: " << ii << " -> mom = " << mom0 << ", " << mom1 << "\n";
    }

    for (size_t ii(0); ii < ngbxs; ++ii)
    {
      Kokkos::Timer kokkostimer;
      const auto moms = calc_massmoments(supers);
      const double ttot(kokkostimer.seconds());

      std::cout << "ii: " << ii << " -> mom = "
                << moms.at(0) << ", " << moms.at(1) << "\n";

      std::cout << "-----\n Total Program Duration: "
                << ttot << "s \n-----\n";
    }
  }
  Kokkos::finalize();
}