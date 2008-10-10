####################################################################
# The Falcon Programming language
#
# CMake configuration file for SDL
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)
cmake_minimum_required(VERSION 2.4.6)
PROJECT(Falcon_SDL)

# Check for environment
INCLUDE( "falmod.cmake" )

# REQUIRED does not work in CMake <=2.4.6 for SDL
Find_Package ( SDL REQUIRED )

# Workaround for the non-working REQUIRED flag
if ( NOT SDL_FOUND )
   message ( FATAL_ERROR "SDL not found!" )
endif ( NOT SDL_FOUND )

INCLUDE_DIRECTORIES( ${SDL_INCLUDE_DIR} )

# Project directories
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(ttf)
ADD_SUBDIRECTORY(image)
ADD_SUBDIRECTORY(mixer)
