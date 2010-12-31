####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for bufext
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE bufext )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)
# Enable this include if the project has a private include
#include_directories( ${PROJECT_SOURCE_DIR}/include )


# Sources files the module is built from.
set(SRC_FILES 
  bufext.cpp
  bufext_ext.cpp
  bufext_st.cpp
  # add more source files starting here:
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  bufext_ext.h
  bufext_st.h
  bytebuf.h
  bitbuf.h
  endianswap.h
  buffererror.h
)


# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)


falcon_install_module( ${CURRENT_MODULE} )
