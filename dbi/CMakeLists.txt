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

falcon_define_module( FALCON_MODULE dbi )
Message( "Adding ${FALCON_MODULE}" )

# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE
   dbi.cpp
   dbi_ext.cpp
   dbi_st.cpp
   dbi_service.cpp
   dbiloaderimpl.cpp
)

#Link
TARGET_LINK_LIBRARIES( ${FALCON_MODULE}  dbi_common falcon_engine )
falcon_install_module( ${FALCON_MODULE}  )
