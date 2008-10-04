####################################################################
# The Falcon Programming language
#
# CMake configuration file for SDL
####################################################################
cmake_minimum_required(VERSION 2.4)
PROJECT(Falcon_SDL)

# Check for environment
INCLUDE( "falmod.cmake" )

# Project directories
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(ttf)
ADD_SUBDIRECTORY(image)
