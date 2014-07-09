####################################################################
# The Falcon Programming language
#
# CMake configuration file for module cmdline
####################################################################

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set( SYS_RC cmdline.rc )
endif()

set(SOURCES
  cmdline.cpp
  cmdline_fm.cpp  
  cmdline_fm.h
  
  ${SYS_RC}
)

FalconDeclareFeather( "cmdline" "${SOURCES}" "" )
