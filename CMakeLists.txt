####################################################################
# The Falcon Programming language
#
# CMake configuration file for DBI
####################################################################

PROJECT(Falcon_DBI)
OPTION(WITH_MYSQL  "Include MySQL DBI Module" OFF)
OPTION(WITH_PGSQL  "Include PostgreSQL DBI Module" OFF)
OPTION(WITH_SQLITE "Include SQLite3 DBI Module" ON)

INCLUDE( "falmod.cmake" )

INCLUDE_DIRECTORIES(BEFORE include)

# Project directories
ADD_SUBDIRECTORY(dbi)

# DBI Drivers
IF(WITH_MYSQL)
  ADD_SUBDIRECTORY(mysql)
ENDIF(WITH_MYSQL)

IF(WITH_PGSQL)
  ADD_SUBDIRECTORY(pgsql)
ENDIF(WITH_PGSQL)

IF(WITH_SQLITE)
  ADD_SUBDIRECTORY(sqlite3)
ENDIF(WITH_SQLITE)
