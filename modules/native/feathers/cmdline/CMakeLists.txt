####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################

if(WIN32 AND NOT MINGW)
   set( SYS_RC cmdline.rc )
endif(WIN32)

add_library(cmdline_fm MODULE
  parser_ext.cpp
  parser_ext.h
  modcmdlineparser.h
  cmdline.cpp
  ${SYS_RC}
)

falcon_finalize_module( cmdline_fm )
 
set(_project_targets ${_project_targets} cmdline_fm PARENT_SCOPE) 