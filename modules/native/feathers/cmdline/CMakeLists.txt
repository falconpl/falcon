####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################

if(WIN32 AND NOT MINGW)
   set( SYS_RC cmdline.rc )
endif()

add_library(cmdline_fm MODULE
  parser_ext.cpp
  parser_ext.h
  module_cmdline.h
  cmdline.cpp
  ${SYS_RC}
)

falcon_finalize_module( cmdline_fm )
 
set(_project_targets ${_project_targets} cmdline_fm PARENT_SCOPE) 