####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################

if(UNIX)
   set(SYS_SPECIFIC process_mod_posix.cpp)
else()
   set(SYS_SPECIFIC process_mod_win.cpp process.rc)
   set_source_files_properties(sys/process_win.cpp PROPERTIES
      COMPILE_DEFINITIONS "UNICODE"
   )
endif()

add_library(process_fm MODULE
   process.cpp
   process_ext.cpp
   process_mod.cpp
   
   process.h
   process_ext.h
   process_mod.h
   ${SYS_SPECIFIC}
)

falcon_finalize_module( process_fm )

set(_project_targets ${_project_targets} process_fm PARENT_SCOPE)
