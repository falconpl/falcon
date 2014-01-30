####################################################################
# The Falcon Programming language
#
# CMake configuration file for module pgsql
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FALCON_DEFINE_MODULE( FALCON_MODULE pgsql )

# Find PgSQL
SET(PGSQL_INCLUDE_SEARCH_PATH
  ${PGSQL_INCLUDE_DIR}
  /usr/include/pgsql
  /usr/local/include/pgsql
  /usr/include/postgresql
)
FIND_PATH(PGSQL_REAL_INCLUDE_DIR libpq-fe.h ${PGSQL_INCLUDE_SEARCH_PATH})

# Find PgSQL Client Library
SET(PGSQL_LIB_SEARCH_PATH ${PGSQL_LIB_DIR} /usr/lib /usr/lib/pgsql /usr/local/lib/pgsql)
SET(PGSQL_LIB_NAMES pq libpq)
FIND_LIBRARY(PGSQL_LIBRARY
  NAMES ${PGSQL_LIB_NAMES}
  PATHS ${PGSQL_LIB_SEARCH_PATH}
)

IF (PGSQL_REAL_INCLUDE_DIR AND PGSQL_LIBRARY)
  INCLUDE_DIRECTORIES(BEFORE ${PGSQL_REAL_INCLUDE_DIR})
ELSE(PGSQL_REAL_INCLUDE_DIR AND PGSQL_LIBRARY)
  IF(NOT PGSQL_REAL_INCLUDE_DIR)
    MESSAGE(SEND_ERROR "PgSQL include path was not found")
    MESSAGE(SEND_ERROR "Looked in: ${PGSQL_INCLUDE_SEARCH_PATH}")
  ENDIF(NOT PGSQL_REAL_INCLUDE_DIR)
  IF(NOT PGSQL_LIBRARY)
    MESSAGE(SEND_ERROR "PgSQL library was not found")
    MESSAGE(SEND_ERROR "Looked for: ${PGSQL_LIB_NAMES}")
    MESSAGE(SEND_ERROR "Looked in: ${PGSQL_LIB_SEARCH_PATH}")
  ENDIF(NOT PGSQL_LIBRARY)
  MESSAGE(FATAL_ERROR "Could not find PgSQL, cannot continue")
ENDIF (PGSQL_REAL_INCLUDE_DIR AND PGSQL_LIBRARY)

MESSAGE(STATUS "Found libpq-fe.h in ${PGSQL_REAL_INCLUDE_DIR}")
MESSAGE(STATUS "Found pgsql client library, ${PGSQL_LIBRARY}")

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)

# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE
   ${dbi_common_files}
   pgsql_fm.cpp
   pgsql_ext.cpp
   pgsql_mod.cpp
)

# Link
IF(MSVC)
  SET(PGSQL_OPT_LIBS wsock32)
ENDIF(MSVC)

TARGET_LINK_LIBRARIES( ${FALCON_MODULE} falcon_engine  ${PGSQL_OPT_LIBS} ${PGSQL_LIBRARY})
FALCON_INSTALL_MODULE2( ${FALCON_MODULE} dbi )
set_target_properties( ${FALCON_MODULE}
	  PROPERTIES 
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Falcon_MOD_DIR}/dbi" )
 