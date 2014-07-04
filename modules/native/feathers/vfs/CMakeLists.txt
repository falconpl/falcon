####################################################################
# The Falcon Programming language
#
# CMake configuration file for module vfs (Virtual File System)
####################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

#Specific system files
if(WIN32 AND NOT MINGW)
   set( SYS_RC vfs.rc )
endif()

add_library(vfs_fm MODULE
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

falcon_finalize_module( vfs_fm )
 
set(_project_targets ${_project_targets} vfs_fm PARENT_SCOPE)
