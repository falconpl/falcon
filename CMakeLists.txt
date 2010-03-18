####################################################################
# The Falcon Programming language
#
# CMake configuration file for DBI
####################################################################
cmake_minimum_required(VERSION 2.6.2)
PROJECT(Falcon_DBI)

# The required package
find_package(Falcon
   HINTS "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
         "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
         "$ENV{FALCON_BIN_PATH}/cmake"
   REQUIRED)

# Options
OPTION(WITH_MYSQL  "Include MySQL DBI Module" OFF)
OPTION(WITH_PGSQL  "Include PostgreSQL DBI Module" OFF)
OPTION(WITH_SQLITE "Include SQLite3 DBI Module" ON)
OPTION(WITH_ORACLE "Include Oracle DBI Module" OFF)

#For now, activate ODBC on windows
IF(MSVC)
   OPTION(WITH_ODBC "Include ODBC DBI Module" ON)
ELSE(MSVC)
   OPTION(WITH_ODBC "Include ODBC DBI Module" OFF)
ENDIF(MSVC)


INCLUDE_DIRECTORIES(BEFORE include)
INCLUDE_DIRECTORIES(${falcon_INCLUDE_DIRS})

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

IF(WITH_ODBC)
  ADD_SUBDIRECTORY(odbc)
ENDIF(WITH_ODBC)

IF(WITH_ORACLE)
  ADD_SUBDIRECTORY(oracle)
ENDIF(WITH_ORACLE)
