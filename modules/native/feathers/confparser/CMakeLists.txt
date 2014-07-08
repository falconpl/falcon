####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################

if(WIN32 AND NOT MINGW)
   set( SYS_RC confparser.rc )
endif()

add_library(confparser_fm MODULE
  confparser.cpp
  confparser_fm.cpp
  confparser_ext.cpp
  confparser_mod.cpp
  
  confparser_ext.h
  confparser_mod.h
  confparser_fm.h
  ${SYS_RC}
)

falcon_finalize_module( confparser_fm )
 
set(_project_targets ${_project_targets} confparser_fm PARENT_SCOPE) 