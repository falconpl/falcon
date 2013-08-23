####################################################################
# The Falcon Programming language
#
# CMake configuration file for module "logging"
####################################################################


if(WIN32)
   set( SYS_RC logging.rc )
   set( SYS_SRC logchannel_syslog_win.cpp  )
else()
   set( SYS_SRC logchannel_syslog_posix.cpp )
endif()

# Target
add_library(logging_fm MODULE
   logging.cpp
   logging_ext.cpp
   logarea.cpp
   logchannel.cpp
   logchannel_engine.cpp
   logchannel_tw.cpp
   logchannel_files.cpp
   logchannel_syslog.cpp
   
   logging_ext.h
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
