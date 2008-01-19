####################################################################
# The Falcon Programming language
#
# CMake configuration file for DBI
####################################################################

PROJECT(Falcon_DBI)
OPTION(WITH_MYSQL  "Include MySQL DBI Module" OFF)
OPTION(WITH_PGSQL  "Include PostgreSQL DBI Module" OFF)
OPTION(WITH_SQLITE "Include SQLite3 DBI Module" ON)

SET(FALCON_PROJECT_BASE "modules/dbi")

# Check for environment
IF($ENV{FALCON_DEVEL_TREE} STREQUAL "" )
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF($ENV{FALCON_DEVEL_TREE} STREQUAL "")

# Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")
# Set common include directory
FALCON_INCLUDE_SETTINGS()

INCLUDE_DIRECTORIES(BEFORE include)

# Project directories
FALCON_ADD_SUBDIRECTORY(dbi)

# DBI Drivers
IF(WITH_MYSQL)
  FALCON_ADD_SUBDIRECTORY(mysql)
ENDIF(WITH_MYSQL)

IF(WITH_PGSQL)
  FALCON_ADD_SUBDIRECTORY(pgsql)
ENDIF(WITH_PGSQL)

IF(WITH_SQLITE)
  FALCON_ADD_SUBDIRECTORY(sqlite3)
ENDIF(WITH_SQLITE)
