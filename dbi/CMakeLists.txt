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

FALCON_DEFINE_MODULE( DBI dbi )

# Target
ADD_LIBRARY( ${DBI} MODULE
   dbi.cpp
   dbi_ext.cpp
   dbi_mod.cpp
   dbiloaderimpl.cpp
)

#Link
FALCON_LINK_MODULE( dbi_fm )
