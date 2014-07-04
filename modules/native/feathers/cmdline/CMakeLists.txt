####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################

if(WIN32 AND NOT MINGW)
   set( SYS_RC cmdline.rc )
endif()

add_library(cmdline_fm MODULE
  cmdline.cpp
  cmdline_fm.cpp  
  cmdline_fm.h
  
  ${SYS_RC}
)

falcon_finalize_module( cmdline_fm )
 
set(_project_targets ${_project_targets} cmdline_fm PARENT_SCOPE) 