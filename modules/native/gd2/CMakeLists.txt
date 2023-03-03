####################################################################
# The Falcon Programming Language
#
# CMake configuration file for gd2
####################################################################

PROJECT(gd2)
cmake_minimum_required(VERSION 3.16.3)
MESSAGE(STATUS "Building GD2 module" )


# Set here project-wide options
find_package( Falcon REQUIRED )

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

#process source directory
ADD_SUBDIRECTORY(src)

