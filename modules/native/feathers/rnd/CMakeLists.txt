####################################################################
# The Falcon Programming Language
#
# CMake configuration file for rnd (Simple random)
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE rnd )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set(SYS_RC rnd.rc)
endif()

# Sources files the module is built from.
set(SRC_FILES 
  rnd.cpp
  rnd_fm.cpp
  rnd_ext.cpp
  ${SYS_RC}
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  rnd_ext.h
  rnd_fm.h
)

# Builds and links from the source files
set( SOURCES
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

FalconDeclareFeather( "rnd" "${SOURCES}" "" )

