####################################################################
# The Falcon Programming language
#
# CMake configuration file for DBI
####################################################################
cmake_minimum_required(VERSION 2.6.2)
project(Falcon_DBI)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# The required package
find_package(Falcon REQUIRED)

# Options
OPTION(FALCON_DBI_BUILD_SQLITE "Include SQLite3 DBI Module" ON)

OPTION(FALCON_DBI_BUILD_MYSQL  "Include MySQL DBI Module" OFF)
OPTION(FALCON_DBI_BUILD_PGSQL  "Include PostgreSQL DBI Module" OFF)
OPTION(FALCON_DBI_BUILD_ORACLE "Include Oracle DBI Module" OFF)
OPTION(FALCON_DBI_BUILD_FIREBIRD "Include Firebird DBI Module" OFF)

#For now, activate ODBC on windows
IF(MSVC)
   OPTION(FALCON_DBI_BUILD_ODBC "Include ODBC DBI Module" ON)
ELSE(MSVC)
   OPTION(FALCON_DBI_BUILD_ODBC "Include ODBC DBI Module" OFF)
ENDIF(MSVC)


INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(${Falcon_INCLUDE_DIRS})

#Common files
set( dbi_common_files
   ../dbi_common/dbi_common.cpp
   ../dbi_common/dbi_error.cpp
   ../dbi_common/dbi_handle.cpp
   ../dbi_common/dbi_inbind.cpp
   ../dbi_common/dbi_outbind.cpp
   ../dbi_common/dbi_params.cpp
   ../dbi_common/dbi_recordset.cpp
   ../dbi_common/dbi_stmt.cpp

   # Header files are useful for IDEs
   ../include/falcon/dbi_error.h
   ../include/falcon/dbi_handle.h
   ../include/falcon/dbi_inbind.h
   ../include/falcon/dbi_outbind.h
   ../include/falcon/dbi_params.h
   ../include/falcon/dbi_recordset.h
   ../include/falcon/dbi_stmt.h
)


# Project directories
ADD_SUBDIRECTORY(dbi)

# DBI Drivers
IF(FALCON_DBI_BUILD_MYSQL)
  ADD_SUBDIRECTORY(mysql)
ENDIF()

IF(FALCON_DBI_BUILD_PGSQL)
  ADD_SUBDIRECTORY(pgsql)
ENDIF()

IF(FALCON_DBI_BUILD_SQLITE)
  ADD_SUBDIRECTORY(sqlite3)
ENDIF()

IF(FALCON_DBI_BUILD_ODBC)
  ADD_SUBDIRECTORY(odbc)
ENDIF()

IF(FALCON_DBI_BUILD_ORACLE)
  ADD_SUBDIRECTORY(oracle)
ENDIF()

IF(FALCON_DBI_BUILD_FIREBIRD)
  ADD_SUBDIRECTORY(fbsql)
ENDIF()

