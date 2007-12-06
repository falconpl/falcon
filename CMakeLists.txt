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

#set common include directory
FALCON_INCLUDE_SETTINGS()

FALCON_ADD_SUBDIRECTORY(engine)
FALCON_ADD_SUBDIRECTORY(rtl)
FALCON_ADD_SUBDIRECTORY(clt)

#also performs installation of the include directory
FALCON_ADD_SUBDIRECTORY("include")

#Setting project wide include directories.
MESSAGE( "INCLUDING TARGET DIR  ${TARGET_DIR}" )
