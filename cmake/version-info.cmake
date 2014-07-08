#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/version-info.cmake
#
#   Generation of variables to describe the version of Falcon.
#   -------------------------------------------------------------------
#   Author: Giancarlo Niccolai
#   Begin: Wed, 02 Jul 2014 23:02:17 +0200
#
#   -------------------------------------------------------------------
#   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
#
#   See LICENSE file for licensing details.
#   
#################################################################


include(version.cmake)

if(NOT FALCON_SONAME_AGE)
   # A couple of useful shortcuts
   set(FALCON_SONAME "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}")
   set(FALCON_SONAME_REV "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}")
endif()

#Automatically generated version info for RC scripts and sources
#CMAKE is good at this, let's use this feature
set(FALCON_VERSION_RC   "${FALCON_VERSION_MAJOR}, ${FALCON_VERSION_MINOR}, ${FALCON_VERSION_REVISION}, ${FALCON_VERSION_PATCH}")
set(FALCON_VERSION_ID   "${FALCON_VERSION_MAJOR}.${FALCON_VERSION_MINOR}.${FALCON_VERSION_REVISION}.${FALCON_VERSION_PATCH}")
set(FALCON_ID   "${FALCON_VERSION_MAJOR}.${FALCON_VERSION_MINOR}.${FALCON_VERSION_REVISION}")


set( Falcon_VERSION "${FALCON_VERSION_ID}" )

# Declare we're clear to version.
message(STATUS "Compiling Falcon ${FALCON_VERSION_ID} on ${CMAKE_SYSTEM}" )



