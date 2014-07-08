####################################################################
# The Falcon Programming language
#
# CMake configuration file for module "logging"
####################################################################


if(WIN32)
   if (NOT MINGW)
      set( SYS_RC logging.rc )
   endif()
   set( SYS_SRC logchannel_syslog_win.cpp  )
else()
   set( SYS_SRC logchannel_syslog_posix.cpp )
endif()

# Target
add_library(logging_fm MODULE
   logging.cpp
   logging_fm.cpp
   logarea.cpp
   logchannel.cpp
   logchannel_engine.cpp
   logchannel_tw.cpp
   logchannel_files.cpp
   logchannel_syslog.cpp
   
   logging_fm.h
   logging_mod.h
   logchannel.h
   logchannel_engine.h
   logchannel_tw.h
   logchannel_files.h
   logchannel_syslog.h
   
   ${SYS_SRC}
   ${SYS_RC}
)

falcon_finalize_module( logging_fm )
 
set(_project_targets ${_project_targets} logging_fm PARENT_SCOPE)
