/*
 * ----- CLEO -----
 * File: readbinary.cpp
 * Project: initialise
 * Created Date: Monday 30th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Monday 30th October 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * tools for reding binary initialisation
 * file e.g. for making gridbox maps or
 * SD initial conditions */


#include "readbinary.hpp"

GblMetadata::GblMetadata(std::ifstream &file)
{
  // read 4 unsigned ints are start of binary file
  file.clear();
  file.seekg(0, std::ios::beg);

  std::vector<unsigned int> uints(4, 0);
  binary_into_buffer<unsigned int>(file, uints);

  d0byte = uints.front();
  charbytes = uints.at(1);
  nvars = uints.at(2);
  mbytes_pervar = uints.at(3);

  const unsigned int offset = 4 * sizeof(unsigned int); // offset from start of file to start of metastring
  metastr = read_global_metastring(file, offset);
}

std::string GblMetadata::read_global_metastring(std::ifstream &file,
                                                const int off) const
/* read 'gblmbytes' bytes of file and interpret as string of global metadata
to print to terminal. Return current position in file (after reading) */
{
  file.seekg(off, std::ios::beg);

  const size_t nchars = charbytes / sizeof(char);
  std::string metastr(nchars, ' ');
  file.read(&metastr[0], nchars);
  
  std::cout << "----------------- gridfile global metastring -----------------\n"
            << metastr
            << "\n--------------------------------------------------------------\n";

  return metastr;
}

VarMetadata::VarMetadata(std::ifstream &file, const int off)
{
  file.seekg(off, std::ios::beg);

  std::vector<unsigned int> uints(3, 0);
  binary_into_buffer<unsigned int>(file, uints);
  
  char chars[2];
  file.read(chars, 2 * sizeof(char));
  
  double dbl;
  file.read(reinterpret_cast<char *>(&dbl), sizeof(double));
  
  b0 = uints.front();
  bsize = uints.at(1);
  nvar = uints.back();
  vtype = chars[0];
  units = chars[1];
  scale_factor = dbl;
}

std::ifstream open_binary(std::string_view filename)
/* open binary file for reading or raise error */
{
  std::string filestr = static_cast<std::string>(filename);
  std::cout << "opening binary file: " << filestr << '\n';
  std::ifstream file(filestr, std::ios::in | std::ios::binary);

  if (!file.is_open())
  {
    throw std::invalid_argument("Cannot open " + filestr);
  }

  return file;
}