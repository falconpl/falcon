####################################################################
# The Falcon Programming language
#
# CMake configuration file for common DBI library
####################################################################

cmake_minimum_required(VERSION 2.6.2)

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_library( dbi_common STATIC
   dbi_bind.cpp
   dbi_error.cpp
   dbi_handle.cpp
   dbi_params.cpp
   dbi_recordset.cpp
   dbi_trans.cpp
   
   # Header files are useful for IDEs
   ../include/falcon/dbi_bind.h
   ../include/falcon/dbi_error.h
   ../include/falcon/dbi_handle.h
   ../include/falcon/dbi_params.h
   ../include/falcon/dbi_recordset.h
   ../include/falcon/dbi_trans.h
)
