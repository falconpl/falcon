####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################

if( WIN32 OR APPLE )
   set(_zlib_default ON)
else( WIN32 OR APPLE )
   set(_zlib_default OFF)
endif( WIN32 OR APPLE)
option(FALCON_WITH_INTERNAL_ZLIB "Use internal zlib library" ${_zlib_default})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32)
   set(SYS_SPECIFIC zlib.rc)
endif()

if(FALCON_WITH_INTERNAL_ZLIB)
  set(ZLIB_LIBRARY "")
  set(ZLIB_SOURCE
    adler32.c
    compress.c
    crc32.c
    deflate.c
    infback.c
    inffast.c
    inflate.c
    inftrees.c
    trees.c
    uncompr.c
    zutil.c
    ${SYS_SPECIFIC}
  )
else(FALCON_WITH_INTERNAL_ZLIB)
   find_package(ZLIB REQUIRED)
   include_directories(${ZLIB_INCLUDE_DIR})
endif(FALCON_WITH_INTERNAL_ZLIB)

# Target
add_library(zlib_fm MODULE
   ${ZLIB_SOURCE}
   # ZLib module source
   zlib.cpp
   zlib_ext.cpp
   zlib_fm.cpp
   
   #zlibstream.cpp
)

target_link_libraries(zlib_fm falcon_engine ${ZLIB_LIBRARIES})

falcon_install_module( zlib_fm )
 
set(_project_targets ${_project_targets} zlib_fm PARENT_SCOPE)
