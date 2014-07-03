#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: test-bigendian.cmake
#
#   Check the endianity of the host system
#   -------------------------------------------------------------------
#   Author: Giancarlo Niccolai
#   Begin: Wed, 02 Jul 2014 23:02:17 +0200
#
#   -------------------------------------------------------------------
#   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
#
#   See LICENSE file for licensing details.
#   
#################################################################

include(TestBigEndian)

TEST_BIG_ENDIAN(falcon_big_endian)
if(falcon_big_endian)
   set(FALCON_LITTLE_ENDIAN 0) 
   message(STATUS "Endianity on \"${CMAKE_SYSTEM}\" as BIG ENDIAN" )
else(falcon_big_endian)
   set(FALCON_LITTLE_ENDIAN 1)
   message(STATUS "Endianity on \"${CMAKE_SYSTEM}\" as LITTLE ENDIAN" )
endif(falcon_big_endian)

