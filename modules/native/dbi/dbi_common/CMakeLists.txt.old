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
   dbi_common.cpp
   dbi_error.cpp
   dbi_handle.cpp
   dbi_inbind.cpp
   dbi_outbind.cpp
   dbi_params.cpp
   dbi_recordset.cpp
   dbi_stmt.cpp
   
   # Header files are useful for IDEs
   ../include/falcon/dbi_error.h
   ../include/falcon/dbi_handle.h
   ../include/falcon/dbi_inbind.h
   ../include/falcon/dbi_outbind.h
   ../include/falcon/dbi_params.h
   ../include/falcon/dbi_recordset.h
   ../include/falcon/dbi_stmt.h
)
