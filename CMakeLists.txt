####################################################################
# The Falcon Programming language
#
# CMake configuration file for SQL modules
####################################################################

PROJECT(Falcon_DBI)
SET(FALCON_PROJECT_BASE "modules/dbi")

#Check for environment
IF($ENV{FALCON_DEVEL_TREE} STREQUAL "" )
   MESSAGE( FATAL_ERROR "Sorry, Falcon environment not correctly configured" )
ENDIF($ENV{FALCON_DEVEL_TREE} STREQUAL "")

#Include falcon configuration
INCLUDE("$ENV{FALCON_DEVEL_TREE}/config.cmake")
#set common include directory
FALCON_INCLUDE_SETTINGS()

INCLUDE_DIRECTORIES(BEFORE include)


#Project directories

#FALCON_ADD_SUBDIRECTORY(mysql)
#FALCON_ADD_SUBDIRECTORY(pgsql)
#FALCON_ADD_SUBDIRECTORY(sql)
#FALCON_ADD_SUBDIRECTORY(sqlite3)
