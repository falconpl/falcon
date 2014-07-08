####################################################################
# The Falcon Programming language
#
# CMake configuration file for module json
####################################################################

if(WIN32 AND NOT MINGW)
   SET( SYS_RC json.rc )
endif()

add_library(json_fm MODULE
   json_mod.cpp
   json_fm.cpp
   json.cpp
   
   json_mod.h
   json_fm.h
   ${SYS_RC}
)

falcon_finalize_module( json_fm )
 
set(_project_targets ${_project_targets} json_fm PARENT_SCOPE)