######################################################################
# CMake file for falcon SHMEM module
#


# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE shmem )


#######################################################################
# Targets
#

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)

if(WIN32)
   set( _files_sys 
        sharedmem_win.cpp
   )
else()
   set( _files_sys 
        sharedmem_posix.cpp
        ipsem_posix.cpp
   )
endif()


# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sources files the module is built from.
set(SRC_FILES 
  ${_files_sys}
  shmem_ext.cpp
  ipsem_ext.cpp
  shared_ipsem.cpp
  shmem.cpp
  
  sharedmem.h
  ipsem.h
  shared_ipsem.h
  
  shmem_ext.h
  version.h
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
)

# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)

falcon_install_module( ${CURRENT_MODULE} )
