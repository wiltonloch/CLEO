/*
 * ----- CLEO -----
 * File: gbxindex.hpp
 * Project: gridboxes
 * Created Date: Tuesday 28th November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Tuesday 28th November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * Functions and structures related to
 * the unique indexes that label CLEO's gridboxes
 */

#ifndef GBXINDEX_HPP
#define GBXINDEX_HPP

struct Gbxindex
/* struct containing gridbox index and its generator struct */
{
  unsigned int value;
  class Gen
  {
  public:
    Gbxindex next() { return {_id++}; }

    Gbxindex next(const unsigned int id)
    {
      return {id};
    }

  private:
    unsigned int _id = 0;
  };
};

#endif // GBXINDEX_HPP