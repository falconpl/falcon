####################################################################
# The Falcon Programming language
#
# CMake configuration file for module vfs (Virtual File System)
####################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

#Specific system files

add_library(vfs_fm MODULE
   vfs_ext.cpp
   classvfs.cpp

   vfs.cpp
)

falcon_finalize_module( vfs_fm )
 
set(_project_targets ${_project_targets} vfs_fm PARENT_SCOPE)
