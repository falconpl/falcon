####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

PROJECT(Falcon_Feathers)
SET(FALCON_PROJECT_BASE "modules/feathers")

#Check for environment
IF($ENV{FALCON_DEVEL_TREE} STREQUAL "" )
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF($ENV{FALCON_DEVEL_TREE} STREQUAL "")

#Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")

# Common for all feathers
INCLUDE_DIRECTORIES(BEFORE ${FALCON_ACTIVE_TREE}/include)
LINK_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/lib )

INCLUDE_DIRECTORIES(BEFORE include)

#Also, include files that may be configured
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)

#And generated libraries
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

#Project directories

FALCON_ADD_SUBDIRECTORY(compiler)
FALCON_ADD_SUBDIRECTORY(confparser)
FALCON_ADD_SUBDIRECTORY(process)
FALCON_ADD_SUBDIRECTORY(regex)
FALCON_ADD_SUBDIRECTORY(socket)
