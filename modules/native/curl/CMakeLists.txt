####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

cmake_minimum_required(VERSION 2.6)
project(curl)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Falcon REQUIRED)
find_package(CURL 7.19.4 REQUIRED)

## process source directories
# add other subdirectories that contain libraries on which the final module depends here..
#add_subdirectory(awesome_library)
# ..before the module's source dir:
add_subdirectory(src)
