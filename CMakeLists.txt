####################################################################
# The Falcon Programming language
#
# CMake configuration file for Core falcon
####################################################################

PROJECT(Falcon_Core)
SET(FALCON_PROJECT_BASE "core")

#Check for environment
IF($ENV{FALCON_DEVEL_TREE} STREQUAL "" )
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF($ENV{FALCON_DEVEL_TREE} STREQUAL "")

#Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")

FALCON_ADD_SUBDIRECTORY(engine)
FALCON_ADD_SUBDIRECTORY(rtl)
FALCON_ADD_SUBDIRECTORY(clt)

#also performs installation of the include directory
FALCON_ADD_SUBDIRECTORY("include")

SET( CONFIG_LIST
   include/falcon/config.h )

FALCON_CONFIG( CONFIG_LIST )
