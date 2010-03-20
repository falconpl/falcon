####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Funcext
####################################################################

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${Falcon_INCLUDE_DIRS}
)

if(WIN32)
   SET( SYS_RC funcext.rc )
endif()

add_library(funcext_fm MODULE
  funcext.cpp
  funcext_ext.cpp
  ${SYS_RC}
)

falcon_finalize_module( funcext_fm )
 
