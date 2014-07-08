####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################

if(UNIX)
   set(SYS_SPECIFIC process_mod_posix.cpp)
else()
   if (NOT MINGW)
      set(SYS_RC process.rc)
   endif()
   set(SYS_SPECIFIC process_mod_win.cpp ${SYS_RC})
   set_source_files_properties(sys/process_win.cpp PROPERTIES
      COMPILE_DEFINITIONS "UNICODE"
   )
endif()

add_library(process_fm MODULE
   process.cpp
   process_fm.cpp
   process_mod.cpp
   
   process.h
   process_fm.h
   process_mod.h
   ${SYS_SPECIFIC}
)

falcon_finalize_module( process_fm )

set(_project_targets ${_project_targets} process_fm PARENT_SCOPE)
