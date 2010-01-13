####################################################################
# The Falcon Programming language
#
# CMake configuration file for module DBI - database interface
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)

falcon_define_module( dbi )
Message( "Adding ${FALCON_MODULE}" )

# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE
   dbi.cpp
   dbi_ext.cpp
   dbi_mod.cpp
   dbiloaderimpl.cpp
)

#Link
falcon_finalize_module( ${FALCON_MODULE} )
