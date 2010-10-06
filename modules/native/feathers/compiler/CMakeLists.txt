####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Compiler
####################################################################

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

set(_project_targets ${_project_targets} compiler_fm PARENT_SCOPE)
