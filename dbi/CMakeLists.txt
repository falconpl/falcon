####################################################################
# The Falcon Programming language
#
# CMake configuration file for module DBI - database interface
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

# Target
ADD_LIBRARY( dbi MODULE
   dbi.cpp
   dbi_ext.cpp
   dbiloaderimpl.cpp
)

#Link
FALCON_LINK_MODULE( dbi )
