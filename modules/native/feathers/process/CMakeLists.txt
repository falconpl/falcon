####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################

if(UNIX)
   SET(SYS_SPECIFIC process_sys_unix.cpp)
else()
   SET(SYS_SPECIFIC process_sys_win.cpp process.rc)
endif()

add_library(process_fm MODULE
   process.cpp
   process_ext.cpp
   process_mod.cpp
   process_st.cpp
   ${SYS_SPECIFIC}
)

falcon_finalize_module( process_fm )

set(_project_targets ${_project_targets} process_fm PARENT_SCOPE) 
