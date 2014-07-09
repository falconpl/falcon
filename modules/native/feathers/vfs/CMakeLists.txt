####################################################################
# The Falcon Programming language
#
# CMake configuration file for module vfs (Virtual File System)
####################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

#Specific system files
if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set( SYS_RC vfs.rc )
endif()

set(SOURCES
   vfs_ext.cpp
   vfs_fm.cpp
   classvfs.cpp
   classfilestat.cpp
   classdirectory.cpp

   vfs_fm.h
   classvfs.h
   classfilestat.h
   classdirectory.h
   
   ${SYS_RC}
   vfs.cpp
)

FalconDeclareFeather( "vfs" "${SOURCES}" "" )