####################################################################
# The Falcon Programming language
#
# CMake configuration file for Core falcon
####################################################################
cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

PROJECT(Falcon_Core)


IF("$ENV{FALCON_DEVEL_TREE}" STREQUAL "")
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF("$ENV{FALCON_DEVEL_TREE}" STREQUAL "")

#Set the default buid type to Debug
IF(NOT CMAKE_BUILD_TYPE)
   SET( CMAKE_BUILD_TYPE $ENV{FALCON_BUILD_TYPE} )

   #Still unset?
   IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
   ENDIF(NOT CMAKE_BUILD_TYPE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# This is the "product version"
SET(FALCON_VERSION_MAJOR      "$ENV{FALCON_VERSION_MAJOR}")
SET(FALCON_VERSION_MINOR      "$ENV{FALCON_VERSION_MINOR}")
SET(FALCON_VERSION_REVISION   "$ENV{FALCON_VERSION_REVISION}")
SET(FALCON_VERSION_PATCH      "$ENV{FALCON_VERSION_PATCH}")

SET(FALCON_VERSION_NAME "$ENV{FALCON_VERSION_NAME}")

#SONAME and soversion (unix so library informations for engine)
# Remember that SONAME never follows project versioning, but
# uses a VERSION, REVISION, AGE format, where
# VERSION: generational version of the project
# REVISION: times this version has been touched
# AGE: Number of version for which binary compatibility is granted
# In eample, 1.12.5 means that this lib may be dynlinked against
# every program using this lib versioned from 1.8 to 1.12.

IF( NOT "$ENV{FALCON_SONAME_AGE}" STREQUAL "" )
   SET(FALCON_SONAME_VERSION $ENV{FALCON_SONAME_VERSION})
   SET(FALCON_SONAME_REVISION $ENV{FALCON_SONAME_REVISION})
   SET(FALCON_SONAME_AGE $ENV{FALCON_SONAME_AGE})

   # A couple of useful shortcuts
   SET(FALCON_SONAME "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}")
   SET(FALCON_SONAME_REV "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}")
ENDIF( NOT "$ENV{FALCON_SONAME_AGE}" STREQUAL "" )

#Automatically generated version info for RC scripts and sources
#CMAKE is good at this, let's use this feature
SET(FALCON_VERSION_RC   "${FALCON_VERSION_MAJOR}, ${FALCON_VERSION_MINOR}, ${FALCON_VERSION_REVISION}, ${FALCON_VERSION_PATCH}")
SET(FALCON_VERSION_ID   "${FALCON_VERSION_MAJOR}.${FALCON_VERSION_MINOR}.${FALCON_VERSION_REVISION}.${FALCON_VERSION_PATCH}")

#Import environment variables
SET(FALCON_BUILD_NAME   $ENV{FALCON_BUILD_NAME})
SET(FALCON_ACTIVE_TREE  $ENV{FALCON_ACTIVE_TREE})
SET(FALCON_SRC_TREE     $ENV{FALCON_SRC_TREE})
SET(FALCON_DEVEL_TREE   $ENV{FALCON_DEVEL_TREE})
SET(FALCON_BUILD_TREE   $ENV{FALCON_BUILD_TREE})


MESSAGE( "Compiling Falcon on ${CMAKE_SYSTEM}" )

##############################################################################
# Including specific build informations
# This is a user-configurable script containing vars
##############################################################################
INCLUDE("${FALCON_DEVEL_TREE}/settings.cmake" OPTIONAL)

#install will happen in FALCON_INSTALL_TREE (With optional FALCON_INSTALL_TREE_LIB)
IF( "${FALCON_INSTALL_TREE}" STREQUAL "" )
   SET( FALCON_INSTALL_TREE  "${FALCON_ACTIVE_TREE}" )
ENDIF( "${FALCON_INSTALL_TREE}" STREQUAL "" )

IF( "${FALCON_INSTALL_TREE_LIB}" STREQUAL "" )
   SET( FALCON_INSTALL_TREE_LIB  "lib" )
ENDIF( "${FALCON_INSTALL_TREE_LIB}" STREQUAL "" )

#sets default installation target dirs.

SET( FALCON_INC_DIR "${FALCON_INSTALL_TREE}/include" )
SET( FALCON_BIN_DIR "${FALCON_INSTALL_TREE}/bin" )
SET( FALCON_LIB_DIR "${FALCON_INSTALL_TREE}/${FALCON_INSTALL_TREE_LIB}" )
IF(WIN32)
   SET( FALCON_MOD_INSTALL "$ENV{FALCON_ACTIVE_TREE}/bin" )
ELSE(WIN32)
   SET( FALCON_MOD_INSTALL "${FALCON_LIB_DIR}/falcon" )
ENDIF(WIN32)

##############################################################################
#  Other defaults
##############################################################################
INCLUDE(TestBigEndian)

MESSAGE( "Testing endianity on ${CMAKE_SYSTEM}" )
TEST_BIG_ENDIAN(falcon_big_endian)
IF(falcon_big_endian)
   SET(FALCON_LITTLE_ENDIAN 0)
ELSE(falcon_big_endian)
   SET(FALCON_LITTLE_ENDIAN 1)
ENDIF(falcon_big_endian)


#IMPORT final destination - the default install directory.
IF("${FALCON_FINAL_DESTINATION}" STREQUAL "" )
   #In windows, we normally install in c:\falcon
   IF(WIN32)
      IF($ENV{PRGORAMS})
         SET(FALCON_FINAL_DESTINATION  "C:\\\\$ENV{PROGRAMS}\\\\falcon" )
      ELSE($ENV{PRGORAMS})
         SET(FALCON_FINAL_DESTINATION  "C:\\\\Program Files\\\\falcon" )
      ENDIF($ENV{PRGORAMS})
   ELSE(WIN32)
      SET(FALCON_FINAL_DESTINATION  "/usr/local" )
   ENDIF(WIN32)
ENDIF("${FALCON_FINAL_DESTINATION}" STREQUAL "" )

IF("${FALCON_FINAL_DESTINATION_LIB}" STREQUAL "" )
   SET(FALCON_FINAL_DESTINATION_LIB  "${FALCON_INSTALL_TREE_LIB}" )
ENDIF("${FALCON_FINAL_DESTINATION_LIB}" STREQUAL "" )

#In windows, we normally install in c:\falcon
IF(WIN32)
   SET( FALCON_HOST_SYSTEM "WINDOWS" )
   SET( FALCON_SYSTEM_WIN 1 )
   #mingw requires -mthreads global option
   IF ( "${CMAKE_GENERATOR}" STREQUAL "MinGW Makefiles" )
      set( CMAKE_EXE_LINKER_FLAGS -mthreads )
      set( CMAKE_SHARED_LINKER_FLAGS -mthreads )
      set( CMAKE_MODULE_LINKER_FLAGS -mthreads )
   ENDIF ( "${CMAKE_GENERATOR}" STREQUAL "MinGW Makefiles" )
ELSE(WIN32)

   #prepare RPATH to the final destination/lib dir
   SET(CMAKE_SKIP_BUILD_RPATH  TRUE)
   SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
   SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   IF ("${FALCON_RPATH}" STREQUAL "")
      SET(CMAKE_INSTALL_RPATH "${FALCON_FINAL_DESTINATION}/${FALCON_FINAL_DESTINATION_LIB}")
   ELSE ("${FALCON_RPATH}" STREQUAL "")
      SET(CMAKE_INSTALL_RPATH "${FALCON_RPATH}")
   ENDIF("${FALCON_RPATH}" STREQUAL "")

   
   # Set common UNIX warnings.
   IF("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")
   ELSE("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")
      add_definitions(-Wall)
   ENDIF("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")

   IF(APPLE)
      SET( FALCON_HOST_SYSTEM "MAC" )
      SET( FALCON_SYSTEM_MAC 1 )
   ELSEIF(UNIX)
      SET( FALCON_HOST_SYSTEM "UNIX" )
      SET( FALCON_SYSTEM_UNIX 1 )
   ELSE(APPLE)
      MESSAGE( "Sorry, can't determine system type" )
   ENDIF(APPLE)
ENDIF(WIN32)


##############################################################################
#  Used macro
##############################################################################

MACRO(FALCON_CONFIG itemlist)
   FOREACH(item ${${itemlist}} )
      MESSAGE( "Configuring ${item}.in into ${item}" )
      CONFIGURE_FILE(
         ${item}.in
         ${item}
         ESCAPE_QUOTES
         @ONLY )
   ENDFOREACH(item)
ENDMACRO(FALCON_CONFIG)

#include also from-source builds to allow compilation from source
INCLUDE_DIRECTORIES(BEFORE ${FALCON_SRC_TREE}/core/include)
INCLUDE_DIRECTORIES( BEFORE ${FALCON_BUILD_TREE}/core/include)
LINK_DIRECTORIES( ${FALCON_BUILD_TREE}/core/engine )

#We'll take cares of prefixes
SET(CMAKE_INSTALL_PREFIX "")

ADD_SUBDIRECTORY(engine)
ADD_SUBDIRECTORY(clt)
ADD_SUBDIRECTORY(devtools)

#also performs installation of the include directory
ADD_SUBDIRECTORY("include")
