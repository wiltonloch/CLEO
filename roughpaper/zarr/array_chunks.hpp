/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: array_chunks.hpp
 * Project: zarr
 * Created Date: Monday 18th March 2024
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 18th March 2024
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Class to manage and write chunks of data to an array in a given memory store.
 */


#ifndef ROUGHPAPER_ZARR_ARRAY_CHUNKS_HPP_
#define ROUGHPAPER_ZARR_ARRAY_CHUNKS_HPP_

#include <vector>
#include <string>
#include <string_view>

#include "./buffer.hpp"

/**
 * @brief A class template for managing and writing chunks of an array.
 *
 * This class provides functionality for writing chunks of an array to a store.
 *
 * @tparam T The type of data elements stored in the buffer.
 */
template <typename T>
class ArrayChunks {
 private:
  std::vector<size_t> chunkshape;   /**< Shape of chunks along each dimension (constant) */
  std::vector<size_t> reducedarray_nchunks;
  /**< Number chunks of array along all but outermost dimension of array (constant) */

  /**
   * @brief Create label for a chunk given current number of chunks written to array.
   *
   * This function creates and converts a vector of integers representing the label of a
   * chunk along each dimension of an array into a string which can be used to name the current
   * chunk that is next to be written to the store.
   *
   * @param totnchunks The total number of chunks written to the array.
   * @return A string representing the label of the current chunk to write.
   */
  std::string chunk_label(const size_t totnchunks) const {
    auto chunk_num = std::vector<size_t>(chunkshape.size(), 0);
    chunk_num.at(0) = totnchunks / vec_product(reducedarray_nchunks);

    for (size_t aa = 1; aa < chunkshape.size(); ++aa) {
      chunk_num.at(aa) = (totnchunks / vec_product(reducedarray_nchunks, aa)) %
        reducedarray_nchunks.at(aa - 1);
    }

    auto chunk_lab = std::string{ "" };
    for (const auto& c : chunk_num) { chunk_lab += std::to_string(c) + "."; }
    chunk_lab.pop_back();   // delete last "."

    return chunk_lab;
  }

 public:
  /**
   * @brief Constructor for the ArrayChunks class.
   *
   * Initializes the ArrayChunks with the provided chunk shape and reduced array shape. Reduced
   * array shape is the shape of the array along all but the outermost dimensions of the array.
   *
   * @param chunkshape The shape of chunks along each dimension.
   * @param reduced_arrayshape The shape of the reduced array along each dimension.
   */
  ChunkWriter(const std::vector<size_t>& chunkshape, const std::vector<size_t>& reduced_arrayshape)
    : chunkshape(chunkshape), reducedarray_nchunks(chunkshape.size() - 1, 0) {

    /* number of dimensions of reduced array is 1 less than actual array ( = array's chunks) */
    assert((reduced_arrayshape.size() == chunkshape.size() - 1) &&
      "reduced array 1 less dimension than array (excludes outermost (0th) dimension");

    /* set number of chunks along all but array's outermost dimension given
    the shape of each chunk and expected shape of final array along those dimensions */
    for (size_t aa = 1; aa < chunkshape.size(); ++aa) {
      /* Assert the chunk size is completely divisible by the array's expected size along that
      dimension in order to ensure good chunking */
      assert((reduced_arrayshape.at(aa - 1) % chunkshape.at(aa) == 0) &&
        "along all but outermost dimension, arrayshape must be completely divisible by chunkshape");
      /* reducedarray_nchunks = number of chunks along all but outermost dimension of array */
      reducedarray_nchunks.push_back(reduced_arrayshape.at(aa - 1) / chunkshape.at(aa));
    }
  }

  /**
   * @brief Gets the shape of a chunk.
   *
   * @return A vector containing the shape (number of data elements) of a chunk
   * along each dimension.
   */
  std::vector<size_t> get_chunkshape() const {
    return chunkshape;
  }

  /**
   * @brief Writes a chunk to the store and increments the total number of chunks written.
   *
   * This function writes the data held in a buffer in the specified store to a chunk identified by
   * "chunk_label" of an array called "name" given the number of chunks of the array already
   * existing. After writing the chunk, the total number of chunks is incremented.
   *
   * @tparam Store The type of the store.
   * @tparam T The type of the data elements stored in the buffer.
   * @param store Reference to the store where the chunk will be written.
   * @param name Name of the array in the store where the chunk will be written.
   * @param totnchunks The total number of chunks of the array already written.
   * @param buffer The buffer containing the data to be written to the chunk.
   * @return The updated total number of chunks after writing.
   */
  template <typename Store, typename T>
  size_t write_chunk(Store& store, const std::string_view name, const size_t totnchunks,
    Buffer<T>& buffer) const {
    buffer.write_buffer_to_chunk(store, name, chunk_label(totnchunks));
    return ++totnchunks;
  }

  /**
   * @brief Writes a chunk to the store and increments the total number of chunks written.
   *
   * This function writes the data stored in the Kokkos view (in host memory) in the specified store
   * to a chunk identified by "chunk_label" of an array called "name" given the number of chunks of
   * the array already existing. After writing the chunk, the total number of chunks is incremented.
   *
   * @tparam Store The type of the store.
   * @tparam T The type of the data elements stored in the buffer.
   * @param store Reference to the store where the chunk will be written.
   * @param name Name of the array in the store where the chunk will be written.
   * @param totnchunks The total number of chunks of the array already written.
   * @param h_data_chunk The view containing the data in host memory to be written to the chunk.
   * @return The updated total number of chunks after writing.
   */
  template <typename Store, typename T>
  size_t write_chunk(Store& store, const std::string_view name, const size_t totnchunks,
    const Buffer<T>::subviewh_buffer h_data_chunk) const {
    store[std::string(name) + '/' + chunk_label(totnchunks)].operator=<T>(h_data_chunk);
    return ++totnchunks;
  }
};

#endif   // ROUGHPAPER_ZARR_ARRAY_CHUNKS_HPP_
