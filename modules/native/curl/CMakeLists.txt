####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

cmake_minimum_required(VERSION 3.16.3)
project(Falcon_curl)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Falcon REQUIRED)
# find_package(CURL 7.18.2 REQUIRED)

## process source directories
# add other subdirectories that contain libraries on which the final module depends here..
add_subdirectory(src)
