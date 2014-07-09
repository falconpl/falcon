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

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set(SYS_SPECIFIC zlib.rc)
else()
	set(SYS_SPECIFIC )
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
set(SOURCES
   ${ZLIB_SOURCE}
   # ZLib module source
   zlib.cpp
   zlib_ext.cpp
   zlib_fm.cpp
   
   #zlibstream.cpp
)

FalconDeclareFeather( "zlib" "${SOURCES}" "" )


