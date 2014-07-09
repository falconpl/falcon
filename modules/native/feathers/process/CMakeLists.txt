####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################

if(UNIX)
   set(SYS_SPECIFIC process_mod_posix.cpp)
else()
   if (NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
      set(SYS_RC process.rc)
   endif()
   set(SYS_SPECIFIC process_mod_win.cpp ${SYS_RC})
   set_source_files_properties(sys/process_win.cpp PROPERTIES
      COMPILE_DEFINITIONS "UNICODE"
   )
endif()

set( SOURCES
   process.cpp
   process_fm.cpp
   process_mod.cpp
   
   process.h
   process_fm.h
   process_mod.h
   ${SYS_SPECIFIC}
)

FalconDeclareFeather( "process" "${SOURCES}" "" )
