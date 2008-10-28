####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib
####################################################################

PROJECT(dynlib)
cmake_minimum_required(VERSION 2.6)

# Set here project-wide options
# OPTION( WITH_OPT  "An poption" OFF)

INCLUDE( "falmod.cmake" )

#process source directory
ADD_SUBDIRECTORY(src)

#add other subdirectories here.
