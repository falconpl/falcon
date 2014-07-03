#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/compilers.cmake
#
#   Seting of compiler-specific options.
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


#
# GCC requires extra flags to get the most nasty errors
#
if( CMAKE_COMPILER_IS_GNUCXX )
   message( "Adding GNU g++ specific flags" )
   add_definitions(-Wall -Wextra -Werror -Wno-error=uninitialized -lpthread)
endif()

if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
   message( "Adding Clang specific flags" )
   add_definitions(-Wall -Wextra -Werror)
endif()

if( APPLE )
	add_definitions(-DUSE_CXX0X)
endif()
