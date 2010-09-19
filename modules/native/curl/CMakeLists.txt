####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

PROJECT(curl)
cmake_minimum_required(VERSION 2.6)

find_package(Falcon REQUIRED)
find_package(CURL REQUIRED)

## process source directories
# add other subdirectories that contain libraries on which the final module depends here..
#add_subdirectory(awesome_library)
# ..before the module's source dir:
add_subdirectory(src)
