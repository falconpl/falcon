####################################################################
# The Falcon Programming Language
#
# CMake configuration file for timestamp (Simple random)
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE timestamp )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32 AND NOT MINGW)
   set( SYS_RC timestamp.rc)
endif()

# Sources files the module is built from.
set(SRC_FILES 
  timestamp.cpp
  timestamp_ext.cpp
  ${SYS_RC}
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  timestamp_ext.h
)

# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)

falcon_install_module( ${CURRENT_MODULE} )

