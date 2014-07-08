####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for bufext
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE bitbuf )

if(WIN32 AND NOT MINGW)
   set( SYS_RC bitbuf.rc )
endif()

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)
# Enable this include if the project has a private include
#include_directories( ${PROJECT_SOURCE_DIR}/include )


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


# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)


falcon_install_module( ${CURRENT_MODULE} )
