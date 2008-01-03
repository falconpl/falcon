####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################
PROJECT(zlib)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

# Target
ADD_LIBRARY( zlib MODULE
   # ZLib source
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

   # ZLib module source
   zlib.cpp
   zlib_ext.cpp
)

#Link
TARGET_LINK_LIBRARIES(zlib falcon_engine)
SET_TARGET_PROPERTIES(zlib
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( zlib )
