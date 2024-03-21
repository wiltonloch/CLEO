# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindNetCDF
----------

Find NetCDF4 library

based on: https://github.com/Kitware/VTK/blob/master/CMake/FindNetCDF.cmake
in general, NetCDF requires C compiler even if only using Fortran

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``NetCDF::NetCDF_C``
  NetCDF C / C++ libraries

``NetCDF::NetCDF_Fortran``
  NetCDF Fortran libraries

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``NetCDF_FOUND``
  NetCDF4 is found (also ``NetCDF_C_FOUND`` and ``NetCDF_Fortran_FOUND``)
``NetCDF_C_LIBRARIES`` and ``NetCDF_Fortran_LIBRARIES
  uncached list of libraries (using full path name) to link against
``NetCDF_C_INCLUDE_DIRS`` and ``NetCDF_Fortran_INCLUDE_DIRS``
  uncached list of libraries (using full path name) to include

Search details:

1. look for CMake-build config files (for C / C++ only)
2. CMake manual search optionally using pkg-config (this step always needed for Fortran, and for C if step 1 fails)

#]=======================================================================]

include(CheckCSourceCompiles)
include(CheckFortranSourceCompiles)

function(netcdf_c)

execute_process(COMMAND nc-config --includedir
  OUTPUT_VARIABLE NC_CONFIG_C_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

find_path(NetCDF_C_INCLUDE_DIR
NAMES netcdf.h
HINTS ${NC_CONFIG_C_INCLUDE_DIR}
DOC "NetCDF C include directory"
)

if(NOT NetCDF_C_INCLUDE_DIR)
  return()
endif()

execute_process(COMMAND nc-config --libdir
  OUTPUT_VARIABLE NC_CONFIG_C_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

find_library(NetCDF_C_LIBRARY
NAMES netcdf
HINTS ${NC_CONFIG_C_LIBDIR}
DOC "NetCDF C library"
)

if(NOT NetCDF_C_LIBRARY)
  return()
endif()

set(CMAKE_REQUIRED_FLAGS)
set(CMAKE_REQUIRED_INCLUDES ${NetCDF_C_INCLUDE_DIR})

set(CMAKE_REQUIRED_LIBRARIES ${NetCDF_C_LIBRARY})
if(ZLIB_FOUND)
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${ZLIB_LIBRARIES})
endif()

list(APPEND CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})

if(UNIX)
  list(APPEND CMAKE_REQUIRED_LIBRARIES m)
endif()

check_c_source_compiles(
[=[
#include <netcdf.h>
#include <stdio.h>

int main(void){
printf("%s", nc_inq_libvers());
return 0;
}
]=]
NetCDF_C_links
)

if(NOT NetCDF_C_links)
  return()
endif()

set(NetCDF_C_FOUND true PARENT_SCOPE)
set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} PARENT_SCOPE)

endfunction(netcdf_c)


function(netcdf_fortran)

execute_process(COMMAND nc-config --fflags
  OUTPUT_VARIABLE NC_CONFIG_Fortran_FLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCHALL "-I([^ ]+)" NC_CONFIG_Fortran_INCLUDE_DIR "${NC_CONFIG_Fortran_FLAGS}")

find_path(NetCDF_Fortran_INCLUDE_DIR
NAMES netcdf.mod
HINTS ${NetCDF_C_INCLUDE_DIR} ${CMAKE_MATCH_1}
DOC "NetCDF Fortran Include"
)

if(NOT NetCDF_Fortran_INCLUDE_DIR)
  return()
endif()

if(CMAKE_VERSION VERSION_LESS 3.20)
  get_filename_component(NetCDF_LIBDIR ${NetCDF_C_LIBRARY} DIRECTORY)
else()
  cmake_path(GET NetCDF_C_LIBRARY PARENT_PATH NetCDF_LIBDIR)
endif()

execute_process(COMMAND nc-config --flibs
  OUTPUT_VARIABLE NC_CONFIG_Fortran_LIBS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCHALL "-L([^ ]+)" NC_CONFIG_Fortran_LIBDIR "${NC_CONFIG_Fortran_LIBS}")

find_library(NetCDF_Fortran_LIBRARY
NAMES netcdff
HINTS ${NetCDF_LIBDIR} ${CMAKE_MATCH_1}
DOC "NetCDF Fortran library"
)

if(NOT NetCDF_Fortran_LIBRARY)
  return()
endif()

set(CMAKE_REQUIRED_FLAGS)
set(CMAKE_REQUIRED_INCLUDES ${NetCDF_Fortran_INCLUDE_DIR})
list(INSERT CMAKE_REQUIRED_LIBRARIES 0 ${NetCDF_Fortran_LIBRARY})

check_fortran_source_compiles(
"program a
use netcdf
implicit none
end program"
NetCDF_Fortran_links
SRC_EXT f90
)

if(NOT NetCDF_Fortran_links)
  return()
endif()

set(NetCDF_Fortran_FOUND true PARENT_SCOPE)

endfunction(netcdf_fortran)

#============================================================
# main program

find_package(ZLIB)
find_package(Threads)
# top scope so can be reused

netcdf_c()

set(_ncdf_req ${NetCDF_C_LIBRARY})

if(Fortran IN_LIST NetCDF_FIND_COMPONENTS)
  netcdf_fortran()
  list(APPEND _ncdf_req ${NetCDF_Fortran_LIBRARY})
endif()

set(CMAKE_REQUIRED_FLAGS)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_LIBRARIES)

mark_as_advanced(NetCDF_C_INCLUDE_DIR NetCDF_Fortran_INCLUDE_DIR NetCDF_C_LIBRARY NetCDF_Fortran_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NetCDF
REQUIRED_VARS _ncdf_req
HANDLE_COMPONENTS
)

if(NetCDF_FOUND)
  set(NetCDF_C_INCLUDE_DIRS ${NetCDF_C_INCLUDE_DIR})
  set(NetCDF_C_LIBRARIES ${NetCDF_C_LIBRARY})

  if(NOT TARGET NetCDF::NetCDF_C)
    add_library(NetCDF::NetCDF_C INTERFACE IMPORTED)
    set_property(TARGET NetCDF::NetCDF_C PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${NetCDF_C_INCLUDE_DIR}")
    set_property(TARGET NetCDF::NetCDF_C PROPERTY INTERFACE_LINK_LIBRARIES "${NetCDF_C_LIBRARY}")

    target_link_libraries(NetCDF::NetCDF_C INTERFACE
    $<$<BOOL:${ZLIB_FOUND}>:${ZLIB_LIBRARIES}>
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    $<$<BOOL:${UNIX}>:m>
    )
  endif()

  if(NetCDF_Fortran_FOUND)
    set(NetCDF_Fortran_INCLUDE_DIRS ${NetCDF_Fortran_INCLUDE_DIR})
    set(NetCDF_Fortran_LIBRARIES ${NetCDF_Fortran_LIBRARY})
    if(NOT TARGET NetCDF::NetCDF_Fortran)
      add_library(NetCDF::NetCDF_Fortran INTERFACE IMPORTED)
      set_property(TARGET NetCDF::NetCDF_Fortran PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${NetCDF_Fortran_INCLUDE_DIR}")
      set_property(TARGET NetCDF::NetCDF_Fortran PROPERTY INTERFACE_LINK_LIBRARIES "${NetCDF_Fortran_LIBRARY}")

      target_link_libraries(NetCDF::NetCDF_Fortran INTERFACE
      $<$<BOOL:${ZLIB_FOUND}>:${ZLIB_LIBRARIES}>
      ${CMAKE_THREAD_LIBS_INIT}
      ${CMAKE_DL_LIBS}
      $<$<BOOL:${UNIX}>:m>
      )
      target_link_libraries(NetCDF::NetCDF_Fortran INTERFACE NetCDF::NetCDF_C)
    endif()
  endif()

endif()

find_program(NetCDF_NCDUMP ncdump
  HINTS "${NC_CONFIG_C_LIBDIR}/.."
  DOC "ncdump program, used for testing")
mark_as_advanced(NetCDF_NCDUMP)
