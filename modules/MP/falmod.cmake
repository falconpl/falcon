####################################################################
# The Falcon Programming Language
#
# Macros and utilities for Falcon modules
####################################################################

#Options common to all the falcon modules
set(INSTDIR "" CACHE STRING "Overrdies the default install path" )

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


#determine falcon installation and set inc/lib paths
IF("$ENV{FALCON_INC_PATH}" STREQUAL "" )

   MESSAGE( "Configuring FALCON using falcon-conf" )

   EXEC_PROGRAM( falcon-conf
         ARGS -i
         OUTPUT_VARIABLE FALCON_INC_PATH )
   MESSAGE( "Read INCLUDE=${FALCON_INC_PATH} from falcon conf" )

   EXEC_PROGRAM( falcon-conf
         ARGS --libs-only-L
         OUTPUT_VARIABLE FALCON_LIB_PATH )

   MESSAGE( "Read LIB=${FALCON_LIB_PATH} from falcon conf" )

   EXEC_PROGRAM( falcon-conf
         ARGS --moddir
         OUTPUT_VARIABLE FALCON_MOD_INSTALL )
   MESSAGE( "Read MOD=${FALCON_MOD_INSTALL} from falcon conf" )

ELSE("$ENV{FALCON_INC_PATH}" STREQUAL "" )
   #Usually, this variables are set in a correctly configured MS-WINDOWS
   #or similar environment to obviate the need for FALCON-CONF
   MESSAGE( "Configuring FALCON from environmental settings" )

   IF ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")
      SET( FALCON_INC_PATH "$ENV{FALCON_INC_PATH}" )
      SET( FALCON_LIB_PATH "$ENV{FALCON_LIB_PATH}" )
      SET( FALCON_MOD_INSTALL "$ENV{FALCON_BIN_PATH}" )
   ELSE ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")
      SET( FALCON_INC_PATH "$ENV{FALCON_ACTIVE_TREE}/include" )
      SET( FALCON_LIB_PATH "$ENV{FALCON_ACTIVE_TREE}/lib" )
      SET( FALCON_MOD_INSTALL "$ENV{FALCON_ACTIVE_TREE}/bin" )
   ENDIF ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")

ENDIF("$ENV{FALCON_INC_PATH}" STREQUAL "" )

#prepare RPATH to the final destination/lib dir
IF ("${FALCON_RPATH}" STREQUAL "")
ELSE ("${FALCON_RPATH}" STREQUAL "")
   SET(CMAKE_SKIP_BUILD_RPATH  TRUE)
   SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
   SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
   SET(CMAKE_INSTALL_RPATH "${FALCON_RPATH}")
ENDIF("${FALCON_RPATH}" STREQUAL "")
   
#Anyhow, override if INSTDIR is given
if(NOT ${INSTDIR} STREQUAL "" )
   SET(FALCON_MOD_INSTALL ${INSTDIR} )
   MESSAGE( "Overriding default install path with ${INSTDIR}" )
endif(NOT ${INSTDIR} STREQUAL "" )

MACRO(FALCON_CLEANUP tgt)
   IF(FALCON_STRIP_TARGET)
      INSTALL( CODE "EXECUTE_PROCESS( COMMAND stirp --strip-unneeded ${FALCON_MOD_INSTALL}/${tgt}.${CMAKE_SHARED_LIBRARY_SUFFIX} )" )
   ENDIF(FALCON_STRIP_TARGET)
ENDMACRO(FALCON_CLEANUP)

# creates the standard name for the falcon module.
MACRO( FALCON_DEFINE_MODULE varname modname )
   SET( ${varname} ${modname}_fm )
ENDMACRO(FALCON_DEFINE_MODULE)

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
   FALCON_CLEANUP( ${tgt} )
ENDMACRO(FALCON_INSTALL_MODULE)
