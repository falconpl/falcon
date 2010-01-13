####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Compiler
####################################################################

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${falcon_INCLUDE_DIRS}
)

if(WIN32)
   set( SYS_RC compiler.rc )
endif()

add_library(compiler_fm MODULE
  compiler.cpp
  compiler_ext.cpp
  compiler_mod.cpp
  compiler_st.cpp
  ${SYS_RC}
)

falcon_finalize_module( compiler_fm )
 
