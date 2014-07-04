####################################################################
# The Falcon Programming Language
#
# CMake configuration file for math_extra
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE math )

if(WIN32 AND NOT MINGW)
   set( SYS_RC math.rc )
endif()

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sources files the module is built from.
set(SRC_FILES 
  math.cpp
  math_fm.cpp
  math_ext.cpp
  ${SYS_RC}
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  math_ext.h
  math_fm.h
)

# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)

falcon_install_module( ${CURRENT_MODULE} )

