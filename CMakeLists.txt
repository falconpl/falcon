####################################################################
# The Falcon Programming language
#
# CMake configuration file for SDL
####################################################################

PROJECT(Falcon_SDL)
SET(FALCON_PROJECT_BASE "modules/sdl")

# Check for environment
IF($ENV{FALCON_DEVEL_TREE} STREQUAL "" )
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF($ENV{FALCON_DEVEL_TREE} STREQUAL "")

# Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")
# Set common include directory
FALCON_INCLUDE_SETTINGS()

# Project directories
FALCON_ADD_SUBDIRECTORY(src)
