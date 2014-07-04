####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################

if(NOT FALCON_VSF_BUILD_ZIP )
	message( STATUS "NOT building VFS zip")
	return()
endif()

message( STATUS "Building VFS subsystem zip")


if( WIN32 OR APPLE )
   set(_minizip_default ON)
else( WIN32 OR APPLE )
   set(_minizip_default OFF)
endif( WIN32 OR APPLE)

option(FALCON_BUILD_VSF_ZIP_INTERNAL "Use internal minizip library" ${_minizip_default})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32)
#   set(SYS_SPECIFIC zip.rc)
   set(SYS_SPECIFIC )
endif()


if(FALCON_WITH_INTERNAL_MINIZIP)
  set(MINIZIP_LIBRARY "")
  set(MINIZIP_SOURCE
  adler32.c
  compress.c
  crc32.c
  deflate.c
  infback.c
  inffast.c
  inflate.c
  inftrees.c
  ioapi.c
  iowin32.c 
  trees.c
  uncompr.c
  unzip.c
  zip.c
  zutil.c
  ${SYS_SPECIFIC}
  )
else(FALCON_WITH_INTERNAL_MINIZIP)
   find_package(MINIZIP REQUIRED)
   include_directories(${MINIZIP_INCLUDE_DIR})
   find_package(ZLIB REQUIRED)
   include_directories(${ZLIB_INCLUDE_DIR})
endif(FALCON_WITH_INTERNAL_MINIZIP)

# Target
add_library(zip_fm MODULE
   ${MINIZIP_SOURCE}
   # ZLib module source
   #zip.cpp
   #zip_ext.cpp
   #zip_st.cpp
   zipstream.cpp
)

target_link_libraries(zip_fm falcon_engine ${MINIZIP_LIBRARIES})

#falcon_install_module( zip_fm )
 
set(_project_targets ${_project_targets} zip_fm PARENT_SCOPE)
