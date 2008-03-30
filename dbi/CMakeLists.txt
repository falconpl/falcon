####################################################################
# The Falcon Programming language
#
# CMake configuration file for module DBI - database interface
####################################################################
PROJECT(dbi)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )

# Target
ADD_LIBRARY( dbi MODULE
   dbi.cpp
   dbi_ext.cpp
   dbiloaderimpl.cpp
)

#Link
TARGET_LINK_LIBRARIES(dbi falcon_engine)
SET_TARGET_PROPERTIES(dbi
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( dbi )
