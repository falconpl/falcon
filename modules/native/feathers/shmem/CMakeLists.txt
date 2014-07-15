######################################################################
# CMake file for falcon SHMEM module
#


#######################################################################
# Targets
#

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)

if(WIN32)
   if(NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   	 set(SYS_RC shmem.rc )
   endif()
   set( _files_sys 
        sharedmem_win.cpp
        ipsem_win.cpp
        sharedmem_private_win.h
        ${SYS_RC}
   )
   set( SYS_LIBS )
elseif(APPLE)
   set( _files_sys 
        sharedmem_posix.cpp
        ipsem_darwin.cpp
        sharedmem_private_posix.h        
   )
   
   set( SYS_LIBS rt)
else()
   set( _files_sys 
        sharedmem_posix.cpp
        ipsem_posix.cpp
        sharedmem_private_posix.h        
   )
   
   set( SYS_LIBS rt)
endif()


# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sources files the module is built from.
set(SRC_FILES 
  ${_files_sys}
  sharedmem.cpp
  shmem_fm.cpp
  shmem_ext.cpp
  ipsem_ext.cpp
  shared_ipsem.cpp
  shmem.cpp
  session.cpp
  session_ext.cpp
  session_srv.cpp    
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES

  shmem_fm.h
  sharedmem.h
  session.h
  ipsem.h
  shared_ipsem.h
  session_srv.h  
  
  shmem_ext.h
  session_ext.h
  version.h

)

# Builds and links from the source files
set( SOURCES
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

FalconDeclareFeather( "shmem" "${SOURCES}" "${SYS_LIBS}" )
