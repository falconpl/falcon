####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set( SYS_RC confparser.rc )
endif()

set(SOURCES
  confparser.cpp
  confparser_fm.cpp
  confparser_ext.cpp
  confparser_mod.cpp
  
  confparser_ext.h
  confparser_mod.h
  confparser_fm.h
  ${SYS_RC}
)

FalconDeclareFeather( "confparser" "${SOURCES}" "" )
