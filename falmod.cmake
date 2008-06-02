####################################################################
# The Falcon Programming language
#
# DBI - Macros and utilities for Falcon modules
####################################################################

#Set the default buid type to RelWithDebInfo
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)


#determine falcon installation and set inc/lib paths
IF("$ENV{FALCON_INC_PATH}" STREQUAL "" )

   MESSAGE( "Configuring FALCON using falcon-conf" )

   EXEC_PROGRAM( falcon-conf
         ARGS -i
         OUTPUT_VARIABLE FALCON_INC_DIR )
   MESSAGE( "Read INCLUDE=${FALCON_INC_DIR} from falcon conf" )

   EXEC_PROGRAM( falcon-conf
         ARGS --libs-only-L
         OUTPUT_VARIABLE FALCON_LIB_DIR )

   MESSAGE( "Read LIB=${FALCON_LIB_DIR} from falcon conf" )

   EXEC_PROGRAM( falcon-conf
         ARGS --moddir
         OUTPUT_VARIABLE FALCON_MOD_INSTALL )
   MESSAGE( "Read MOD=${FALCON_MOD_INSTALL} from falcon conf" )

ELSE("$ENV{FALCON_INC_PATH}" STREQUAL "" )
   #Usually, this variables are set in a correctly configured MS-WINDOWS
   #or similar environment to obviate the need for FALCON-CONF
   MESSAGE( "Configuring FALCON from environmental settings" )

   SET( FALCON_INC_PATH $ENV{FALCON_INC_PATH} CACHE )
   SET( FALCON_LIB_PATH $ENV{FALCON_LIB_PATH} CACHE )
   SET( FALCON_MOD_INSTALL $ENV{FALCON_BIN_PATH} CACHE )

ENDIF("$ENV{FALCON_INC_PATH}" STREQUAL "" )

MACRO(FALCON_CLEANUP tgt)
   IF(FALCON_STRIP_TARGET)
      INSTALL( CODE "EXECUTE_PROCESS( COMMAND stirp --strip-unneeded ${FALCON_MOD_INSTALL}/${tgt}.${CMAKE_SHARED_LIBRARY_SUFFIX} )" )
   ENDIF(FALCON_STRIP_TARGET)
ENDMACRO(FALCON_CLEANUP)


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
   FALCON_CLEANUP( ${tgt} )
ENDMACRO(FALCON_INSTALL_MODULE)
