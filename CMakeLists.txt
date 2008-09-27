####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################
cmake_minimum_required(VERSION 2.4)
PROJECT(Falcon_Feathers)

#Check for environment
IF("$ENV{FALCON_DEVEL_TREE}" STREQUAL "" )
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF($ENV{FALCON_DEVEL_TREE} STREQUAL "")

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

#install will happen in FALCON_INSTALL_TREE (With optional FALCON_INSTALL_TREE_LIB)
IF( "${FALCON_INSTALL_TREE}" STREQUAL "" )
   SET( FALCON_INSTALL_TREE  "$ENV{FALCON_ACTIVE_TREE}" )
ENDIF( "${FALCON_INSTALL_TREE}" STREQUAL "" )

IF( "${FALCON_INSTALL_TREE_LIB}" STREQUAL "" )
   SET( FALCON_INSTALL_TREE_LIB  "lib" )
ENDIF( "${FALCON_INSTALL_TREE_LIB}" STREQUAL "" )


#This is needed only when we don't install the core before starting feathers.
#This usually happens in external builds (i.e. build.sh for core and feathers)
SET( FALCON_INC_DIR "$ENV{FALCON_SRC_TREE}/core/include" ) # < for non config stuff

SET( FALCON_BIN_DIR "${FALCON_INSTALL_TREE}/bin" )
SET( FALCON_LIB_DIR "${FALCON_INSTALL_TREE}/${FALCON_INSTALL_TREE_LIB}" )
IF( WIN32 )
   SET( FALCON_MOD_INSTALL "${FALCON_BIN_DIR}" )
ELSE( WIN32 )
   # Set common UNIX warnings.
   add_definitions(-Wall)
   
   SET( FALCON_MOD_INSTALL "${FALCON_LIB_DIR}/falcon" )
ENDIF( WIN32 )

MACRO(FALCON_LINK_MODULE tgt )
   TARGET_LINK_LIBRARIES(${tgt} falcon_engine)
   FALCON_INSTALL_MODULE( ${tgt} )
ENDMACRO(FALCON_LINK_MODULE)

MACRO(FALCON_INSTALL_MODULE tgt )
   IF(APPLE)
      SET_TARGET_PROPERTIES(${tgt}
         PROPERTIES 
		    PREFIX ""
		    SUFFIX ".dylib" )
   ELSE(APPLE)
      SET_TARGET_PROPERTIES(${tgt}
         PROPERTIES PREFIX "")
   ENDIF(APPLE)
   #Install
   INSTALL( TARGETS ${tgt}
            DESTINATION ${FALCON_MOD_INSTALL} )

ENDMACRO(FALCON_INSTALL_MODULE)

#Projects settings
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_DIR}")
INCLUDE_DIRECTORIES(BEFORE "$ENV{FALCON_BUILD_TREE}/core/include") # < for config stuff

IF( WIN32 )
   LINK_DIRECTORIES("${FALCON_BIN_DIR}")
ELSE( WIN32 )
   LINK_DIRECTORIES( "${FALCON_LIB_DIR}" )
ENDIF( WIN32 )

#Project directories

ADD_SUBDIRECTORY(compiler)
ADD_SUBDIRECTORY(confparser)
ADD_SUBDIRECTORY(funcext)
ADD_SUBDIRECTORY(mxml)
ADD_SUBDIRECTORY(process)
ADD_SUBDIRECTORY(regex)
ADD_SUBDIRECTORY(socket)
ADD_SUBDIRECTORY(zlib)
