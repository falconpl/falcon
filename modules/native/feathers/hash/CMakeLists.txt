####################################################################
# The Falcon Programming Language
#
# CMake configuration file for hash
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE hash )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)
# Enable this include if the project has a private include
#include_directories( ${PROJECT_SOURCE_DIR}/include )

if(WIN32 AND NOT MINGW)
   set( SYS_RC hash.rc )
endif()

# Sources files the module is built from.
set(SRC_FILES 
  hash.cpp
  hash_mod.cpp
  hash_fm.cpp
  # add more source files starting here:
  adler32.cpp
  md2.cpp
  md4.cpp
  md5.cpp
  ripemd.cpp
  sha1.cpp
  sha256_sha224.cpp
  sha512_sha384.cpp
  tiger.cpp
  tiger_sboxes.cpp
  whirlpool.cpp
  ${SYS_RC}
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  hash_fm.h
  hash_mod.h
  hash_defs.h 
  # add more header files starting here:
  adler32.h
  sha1.h
  sha256_sha224.h
  sha512_sha384.h
  md2.h
  md4.h
  md5.h
  tiger.h
  whirlpool.h
)


# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)


falcon_install_module( ${CURRENT_MODULE} )

set(_project_targets ${_project_targets} ${CURRENT_MODULE} PARENT_SCOPE)
