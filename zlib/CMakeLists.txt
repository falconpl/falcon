####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################
PROJECT(zlib)

INCLUDE(FindZLIB)

OPTION(WITH_INTERNAL_ZLIB "Use internal zlib library" OFF)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )

IF (WITH_INTERNAL_ZLIB)
  SET(ZLIB_LIBRARY "")
  SET(ZLIB_SOURCE
    adler32.c
    compress.c
    crc32.c
    deflate.c
    gzio.c
    infback.c
    inffast.c
    inflate.c
    inftrees.c
    trees.c
    uncompr.c
    zutil.c
  )
ELSE(WITH_INTERNAL_ZLIB)
   IF(NOT ZLIB_FOUND AND NOT ZLIB_LIBRARIES)
      IF(NOT ZLIB_LIB_NAMES)
         SET(ZLIB_LIB_NAMES libz.so libz.a zlib.dll)
      ENDIF(NOT ZLIB_LIB_NAMES)

      IF(NOT ZLIB_INCLUDE_SEARCH_PATH)
         SET(ZLIB_INCLUDE_SEARCH_PATH /usr/include /usr/local/include )
      ENDIF(NOT ZLIB_INCLUDE_SEARCH_PATH)

      IF(NOT ZLIB_LIB_SEARCH_PATH)
          SET(ZLIB_LIB_SEARCH_PATH /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64)
      ENDIF(NOT ZLIB_LIB_SEARCH_PATH)


      FIND_PATH(ZLIB_INCLUDE_DIR zlib.h ${ZLIB_INCLUDE_SEARCH_PATH})
      FIND_LIBRARY(ZLIB_LIBRARIES
         NAMES ${ZLIB_LIB_NAMES}
         PATHS ${ZLIB_LIB_SEARCH_PATH}
      )
   ENDIF(NOT ZLIB_FOUND AND NOT ZLIB_LIBRARIES)

   IF(ZLIB_INCLUDE_DIR AND ZLIB_LIBRARIES)
      INCLUDE_DIRECTORIES(BEFORE ${ZLIB_REAL_INCLUDE_DIR})
   ELSE(ZLIB_INCLUDE_DIR AND ZLIB_LIBRARIES)
      IF(NOT ZLIB_INCLUDE_DIR)
         MESSAGE(SEND_ERROR "ZLib include path was not found")
         MESSAGE(SEND_ERROR "Looked in: ${ZLIB_INCLUDE_SEARCH_PATH}")
      ENDIF(NOT ZLIB_INCLUDE_DIR)
      IF(NOT ZLIB_LIBRARIES)
         MESSAGE(SEND_ERROR "ZLib library was not found")
         MESSAGE(SEND_ERROR "Looked for: ${ZLIB_LIB_NAMES}")
         MESSAGE(SEND_ERROR "Looked in: ${ZLIB_LIB_SEARCH_PATH}")
      ENDIF(NOT ZLIB_LIBRARIES)
      MESSAGE(SEND_ERROR "Could not find ZLib, you can:")
      MESSAGE(SEND_ERROR "Specify: -DWITH_INTERNAL_ZLIB=ON")
      MESSAGE(SEND_ERROR "Or use:")
      MESSAGE(SEND_ERROR "-DZLIB_INCLUE_DIR=PATH_TO_ZLIB_INCLUDES")
      MESSAGE(FATAL_ERROR "-DZLIB_LIBRARIES=PATH_TO_ZLIB_LIBRARY")
   ENDIF(ZLIB_INCLUDE_DIR AND ZLIB_LIBRARIES)
ENDIF(WITH_INTERNAL_ZLIB)

# Target
ADD_LIBRARY( zlib MODULE
   ${ZLIB_SOURCE}
   # ZLib module source
   zlib.cpp
   zlib_ext.cpp
   #zlibstream.cpp
)

#Link
TARGET_LINK_LIBRARIES(zlib falcon_engine ${ZLIB_LIBRARIES})
SET_TARGET_PROPERTIES(zlib
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( zlib )
