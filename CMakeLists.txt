set(TARGET "merge-constraints")

set(TARGET_SRC
  ${TARGET}.cc
  )

cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_CXX_STANDARD 20)

find_package(deal.II 9.6.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

deal_ii_initialize_cached_variables()
set(CLEAN_UP_FILES *.vtk *.csv)
project(${TARGET})
deal_ii_invoke_autopilot()
