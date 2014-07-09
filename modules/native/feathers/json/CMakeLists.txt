####################################################################
# The Falcon Programming language
#
# CMake configuration file for module json
####################################################################

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   SET( SYS_RC json.rc )
endif()

set( SOURCES
   json_mod.cpp
   json_fm.cpp
   json.cpp
   
   json_mod.h
   json_fm.h
   ${SYS_RC}
)

FalconDeclareFeather( "json" "${SOURCES}" "" )
