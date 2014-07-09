####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Funcext
####################################################################

if(WIN32 AND NOT FALCON_STATIC_FEATHERS)
   SET( SYS_RC funcext.rc )
endif()

add_library(funcext_fm MODULE
  funcext.cpp
  funcext_ext.cpp
  ${SYS_RC}
)

falcon_finalize_module( funcext_fm )
 
set(_project_targets ${_project_targets} funcext_fm PARENT_SCOPE)