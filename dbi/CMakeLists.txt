####################################################################
# The Falcon Programming language
#
# CMake configuration file for module DBI - database interface
####################################################################
PROJECT(dbi)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_DIR}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_DIR}")

# Target
ADD_LIBRARY( dbi MODULE
   dbi.cpp
   dbi_ext.cpp
   dbiloaderimpl.cpp
)

#Link
FALCON_LINK_MODULE( dbi )
