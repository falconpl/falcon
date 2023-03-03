####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for MP
####################################################################

PROJECT(MP)
cmake_minimum_required(VERSION 3.16.3)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
# Set here project-wide options
# OPTION( WITH_OPT  "An poption" OFF)

#message("ARPREC_ROOT=${ARPREC_ROOT}")

find_package( Falcon REQUIRED )
#find_package(ARPREC REQUIRED)
#include_directories(${ARPREC_INCLUDE_DIR})


#process source directory
ADD_SUBDIRECTORY(src)

#add other subdirectories here.
