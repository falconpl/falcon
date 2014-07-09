####################################################################
# The Falcon Programming language
#
# CMake configuration file for module "logging"
####################################################################


if(WIN32)
   if (NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
      set( SYS_RC logging.rc )
   endif()
   set( SYS_SRC logchannel_syslog_win.cpp  )
else()
   set( SYS_SRC logchannel_syslog_posix.cpp )
endif()

# Target
set(SOURCES
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

FalconDeclareFeather( "logging" "${SOURCES}" "" )

