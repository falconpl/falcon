####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

PROJECT(Falcon_Feathers)

#Check for environment
IF($ENV{FALCON_DEVEL_TREE} STREQUAL "" )
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

SET( FALCON_INC_DIR "$ENV{FALCON_ACTIVE_TREE}/include" )
SET( FALCON_LIB_DIR "$ENV{FALCON_ACTIVE_TREE}/lib" )
SET( FALCON_MOD_INSTALL "$ENV{FALCON_ACTIVE_TREE}/bin" )

MACRO(FALCON_LINK_MODULE tgt )
   TARGET_LINK_LIBRARIES(${tgt} falcon_engine)
   FALCON_INSTALL_MODULE( ${tgt} )
ENDMACRO(FALCON_LINK_MODULE)

MACRO(FALCON_INSTALL_MODULE tgt )
   SET_TARGET_PROPERTIES(${tgt}
      PROPERTIES PREFIX "")

   #Install
   INSTALL( TARGETS ${tgt}
            DESTINATION ${FALCON_MOD_INSTALL} )

ENDMACRO(FALCON_INSTALL_MODULE)

#Project directories

ADD_SUBDIRECTORY(compiler)
ADD_SUBDIRECTORY(confparser)
ADD_SUBDIRECTORY(mxml)
ADD_SUBDIRECTORY(process)
ADD_SUBDIRECTORY(regex)
ADD_SUBDIRECTORY(socket)
ADD_SUBDIRECTORY(zlib)
