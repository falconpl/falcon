####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for bufext
####################################################################

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set( SYS_RC bitbuf.rc )
endif()

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)

# Sources files the module is built from.
set(SRC_FILES 
  bitbuf_fm.cpp
  bitbuf_ext.cpp
  bitbuf_mod.cpp
  bitbuf.cpp
  # add more source files starting here:
  ${SYS_RC}
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  bitbuf_ext.h
  bitbuf_mod.h
  buffererror.h
  bitbuf_fm.h
)

set(SOURCES ${SRC_FILES} ${HDR_FILES} )

FalconDeclareFeather( "bitbuf" "${SOURCES}" "" )
 
