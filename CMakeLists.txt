####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib
####################################################################

PROJECT(Falcon_dynlib)
cmake_minimum_required(VERSION 2.6)

# Set here project-wide options
OPTION( WITH_TESTS  "Compile test_dynlib.c to generate the test library" OFF)

INCLUDE( "falmod.cmake" )

#process source directory
ADD_SUBDIRECTORY(src)
