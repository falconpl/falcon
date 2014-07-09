####################################################################
# The Falcon Programming Language
#
# CMake configuration file for hash
####################################################################

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
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
set( SOURCES
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

FalconDeclareFeather( "hash" "${SOURCES}" "" )
