####################################################################
# The Falcon Programming language
#
# CMake configuration file for module PgSQL
####################################################################
PROJECT(pgsql)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
INCLUDE_DIRECTORIES(BEFORE /usr/include/postgresql)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

IF(WIN32)
   SET( SYS_RC pgsql.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( pgsql MODULE
#   pgsql.cpp
#   pgsql_ext.cpp
   pgsql_mod.cpp
#   ${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(pgsql falcon_engine pq)
SET_TARGET_PROPERTIES(pgsql
   PROPERTIES PREFIX "")

#Install
FALCON_MODULE_INSTALL( pgsql )
