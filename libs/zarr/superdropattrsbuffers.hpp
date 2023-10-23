/*
 * ----- CLEO -----
 * File: superdropattrsbuffers.hpp
 * Project: zarr
 * Created Date: Monday 23rd October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 23rd October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * structs obeying the superdropsbuffers concept in order to
 * write out superdrop attributes into a ragged array in a 
 * fsstore via a buffer
 */


#ifndef ATTRSBUFFERS_HPP 
#define ATTRSBUFFERS_HPP 

#include <concepts>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>
#include <tuple>

#include "../cleoconstants.hpp"
#include "./fsstore.hpp"
#include "./storehelpers.hpp"
#include "./superdropsbuffers.hpp"
#include "superdrops/superdrop.hpp"

namespace dlc = dimless_constants;

template <typename T>
struct SuperdropAttrBuffer
/* generic struct satisfies the SuperdropletsBuffer concept.
Can be inherited and then used to put a single attribute called
'attr' into a buffer (given an implementation of the
'copy2buffer' function) and write the accompanying metadata
obeying the zarr storage specificatino version 2.0 via
'writechunk' and 'writemetadata' functions */
{
  const std::string attr;  // name of attribute in fsstore
  const std::string dtype; // datatype stored in arrays
  std::vector<T> buffer;   // buffer to fill before writing to store

  SuperdropAttrBuffer(const std::string attr,
                      const std::string dtype)
      : attr(attr), dtype(dtype), buffer(0) {}

  virtual ~SuperdropAttrBuffer(){};

  virtual std::pair<unsigned int, unsigned int> 
  copy2buffer(const Superdrop &superdrop,
              const unsigned int ndata, const unsigned int j) = 0;
  /* virtual void function placeholding function for
  copying superdrop's data into a buffer vector at j'th index */

  std::pair<unsigned int, unsigned int>
  writechunk(FSStore &store, unsigned int chunkcount)
  /* write buffer vector into attr's store at chunkcount
  and then replace contents of buffer with numeric limit */
  {
    return storagehelper::
        writebuffer2chunk(store, buffer, attr, chunkcount);
  }

  void writejsons(FSStore &store, const SomeMetadata &md) const
  /* write metadata for attr's array into store */
  {
    const std::string metadata = storagehelper::
        metadata(md.zarr_format, md.order, md.shape,
                 md.chunks, dtype, md.compressor,
                 md.fill_value, md.filters);

    const std::string arrayattrs = "{\"_ARRAY_DIMENSIONS\": " + md.dims + "}";

    storagehelper::
        writezarrjsons(store, attr, metadata, arrayattrs);
  }

  void set_buffersize(const size_t maxchunk)
  {
    if (buffer.size() != maxchunk)
    {
      buffer = std::vector<T>(maxchunk, std::numeric_limits<T>::max());
    }
  }
};

struct IdIntoStore : AttributeIntoStoreViaBuffer<size_t>
{
  IdIntoStore()
      : AttributeIntoStoreViaBuffer("sd_id", "<u8"){};

  unsigned int copy2buffer(const Superdrop &superdrop,
                           unsigned int j)
  {
    return storagehelper::
        val2buffer<size_t>(superdrop.sd_id.value, buffer, j);
  }
};

#endif // ATTRSBUFFERS_HPP 
