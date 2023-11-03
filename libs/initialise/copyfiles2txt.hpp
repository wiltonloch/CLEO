/*
 * ----- CLEO -----
 * File: copyfiles2txt.hpp
 * Project: initialise
 * Created Date: Friday 13th October 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * Last Modified: Friday 3rd November 2023
 * Modified By: CB
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * Copyright (c) 2023 MPI-M, Clara Bayley
 * -----
 * File Description:
 * structure for opening files given their filenames and copying their
 * contents line by line into a .txt file. Useful for copying
 * the details of a model setup e.g. configuration and
 * values of constants
 */

#ifndef COPYFILES2TXT_HPP
#define COPYFILES2TXT_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

void copyfiles2txt(const std::string setup_txt,
                       const std::vector<std::string> files2copy);
/* creates new empty file called setup_txt and copies contents
of files listed in files2copy vector one by one */

#endif // COPYFILES2TXT_HPP