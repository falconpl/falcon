######################################################################
# CMake file for mod_falcon
#

MESSAGE( "Creating the Falcon module for Apache 2 (mod_apache)" )

set( APACHE2_LIBEXECDIR "${Falcon_LIB_DIR}" 
   CACHE STRING "Place where mod_falcon will be installed")

######################################################################
# Find APACHE and APR libraries
#

SET(APACHE_INCLUDE_SEARCH_PATH
  ${APACHE_INCLUDE_DIR}
  /usr/include/apache2
  /usr/local/include/apache2
)
FIND_PATH(APACHE_REAL_INCLUDE_DIR httpd.h ${APACHE_INCLUDE_SEARCH_PATH})

MESSAGE( "Configuring APR using pkg-config" )

EXEC_PROGRAM( pkg-config
      ARGS --cflags apr-1 apr-util-1
      OUTPUT_VARIABLE APR_CFLAGS )
MESSAGE( "From pkg-config APR INCLUDE=${APR_CFLAGS}")

EXEC_PROGRAM( pkg-config
      ARGS --libs apr-1 apr-util-1
      OUTPUT_VARIABLE APR_LD_FLAGS )

MESSAGE( "From pkg-config APR LIB=${APR_LD_FLAGS}" )

#Add libcrun on solaris
IF( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
   MESSAGE( "Adding libCRun link in SunOS" )
   SET( SYSLIBS Crun )
ENDIF( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${APACHE_REAL_INCLUDE_DIR}
)


######################################################################
# Add libcrun on solaris
# CMake on solaris doesn't link crun when linking C and C++ modules,
#

IF( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
   MESSAGE( "Adding libCRun link in SunOS" )
   SET( SYSLIBS Crun )
   #SET( CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -lCRun" )
ENDIF( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )


#######################################################################
# Targets
#

set(SRC_FILES
  mod_falcon.c
  apache_errhand.cpp
  apache_reply.cpp
  apache_request.cpp
  apache_stream.cpp
  apache_ll.cpp
  falcon_functions.cpp
  mod_falcon_config.cpp
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  apache_errhand.h
  apache_reply.h
  apache_request.h
  apache_stream.h
  mod_falcon_config.h
  mod_falcon.h
)

ADD_LIBRARY( mod_falcon MODULE
   ${SRC_FILES}
   ${WOPI_SOURCES}
   ${HDR_FILES} # optional but useful, see comment above.
)


tARGET_LINK_LIBRARIES( mod_falcon
   falcon_engine
   ${SYSLIBS}
)

SET_TARGET_PROPERTIES( mod_falcon
   PROPERTIES PREFIX ""
   COMPILE_FLAGS "${APR_CFLAGS}"
   LINK_FLAGS "${APR_LD_FLAGS}"
)


INSTALL(TARGETS mod_falcon
  LIBRARY DESTINATION "${APACHE2_LIBEXECDIR}"
)
