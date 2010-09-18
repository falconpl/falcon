####################################################################
# The Falcon Programming Language
#
# CMake configuration file for gd2
####################################################################

PROJECT(gd2)
cmake_minimum_required(VERSION 2.6)

# Set here project-wide options
# OPTION( WITH_OPT  "An poption" OFF)

#INCLUDE( "falmod.cmake" )
find_package( Falcon REQUIRED )

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

#process source directory
ADD_SUBDIRECTORY(src)

#add other subdirectories here.
