/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: cartesian_decomposition.hpp
 * Project: cartesiandomain
 * Created Date: Tuesday 30th July 2023
 * Author: Wilton J. Loch
 * Additional Contributors:
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Class and suporting functions to perform domain decomposition
 * in a cartesian domain
 */

#ifndef LIBS_CARTESIANDOMAIN_CARTESIAN_DECOMPOSITION_HPP_
#define LIBS_CARTESIANDOMAIN_CARTESIAN_DECOMPOSITION_HPP_

#include <cstddef>
#include <array>
#include <vector>
#include <map>

class CartesianDecomposition {
 private:
    int my_rank;
    // Number of dimensions of the global domain
    std::vector<size_t> ndims;
    // Global origins of all partitions
    std::vector<std::array<size_t, 3>> partition_origins;
    // Sizes of all partitions
    std::vector<std::array<size_t, 3>> partition_sizes;

    std::array<double, 3> partition_begin_coordinates;
    std::array<double, 3> partition_end_coordinates;

    std::array<double, 3> gridbox_size;
    std::array<size_t, 3> dimension_bound_behavior;

    std::map<std::array<int, 3>, int> neighboring_processes;
    std::array<size_t, 3> decomposition;
    // Number of local gridboxes
    size_t total_local_gridboxes;

    void calculate_domain_coordinates();

 public:
    CartesianDecomposition();
    ~CartesianDecomposition();

    // Creates the decomposition
    bool create(std::vector<size_t> ndims,
                double gridbox_z_size,
                double gridbox_x_size,
                double gridbox_y_size);

    size_t get_total_local_gridboxes() const;
    size_t get_total_global_gridboxes() const;
    std::array<size_t, 3> get_local_partition_origin() const;
    std::array<size_t, 3> get_local_partition_size() const;
    int get_gridbox_owner_process(size_t global_gridbox_index) const;
    int get_partition_index_from_slice(std::array<int, 3> slice_indices) const;
    std::array<int, 3> get_slice_indices_from_partition(int partition_index) const;

    void set_gridbox_size(double z_size, double x_size, double y_size);
    void set_dimensions_bound_behavior(std::array<size_t, 3> behaviors);

    int local_to_global_gridbox_index(size_t local_gridbox_index, int process = -1) const;
    int global_to_local_gridbox_index(size_t global_gridbox_index) const;

    // Checks whether a coordinate is bounded by one specific partition
    bool check_indices_inside_partition(std::array<size_t, 3> indices,
                                        int partition_index) const;

    void calculate_neighboring_processes();

    size_t get_local_bounding_gridbox(std::array<double, 3> & coordinates) const;
};



// Given the global domain, a global decomposition and a partition index,
// returns the partition origin and size
void construct_partition(const std::vector<size_t> ndims,
                         std::vector<size_t> decomposition, int partition_index,
                         std::array<size_t, 3> &partition_origin,
                         std::array<size_t, 3> &partition_size);

// Adds all permutations of a particular decomposition and removes the ones that
// do not fit the global dimension sizes
void permute_and_trim_factorizations(std::vector<std::vector<size_t>> &factors,
                                     const std::vector<size_t> ndims);

// Finds the best decomposition given by the most even division of gridboxes among processes
int find_best_decomposition(std::vector<std::vector<size_t>> &factors,
                            const std::vector<size_t> ndims);

size_t get_index_from_coordinates(const std::vector<size_t> &ndims, const size_t k, const size_t i,
                                  const size_t j);
std::array<size_t, 3> get_coordinates_from_index(const std::vector<size_t> &ndims,
                                                 const size_t index);
std::vector<std::vector<size_t>> factorize(int n);
void factorize_helper(int n, int start, std::vector<size_t> &current,
                     std::vector<std::vector<size_t>> &result);
void heap_permutation(std::vector<std::vector<size_t>> &results, std::vector<size_t> arr,
                     int size);
int get_multiplications_to_turn_int(double entry_value);

#endif  // LIBS_CARTESIANDOMAIN_CARTESIAN_DECOMPOSITION_HPP_
