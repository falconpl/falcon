####################################################################
# The Falcon Programming language
#
# CMake configuration file for module pgsql
####################################################################
PROJECT(pgsql)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
# TODO: make configurable in some way
INCLUDE_DIRECTORIES(/usr/include/postgresql)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

# Target
ADD_LIBRARY( pgsql MODULE
   pgsql.cpp
   pgsql_ext.cpp
   pgsql_srv.cpp
)

#Link
TARGET_LINK_LIBRARIES(pgsql falcon_engine pq)
SET_TARGET_PROPERTIES(pgsql
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( pgsql )
