#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/sys_setup-win.cmake
#
#   MS-Windows specific build system setup
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

if(WIN32)

	#mingw requires -mthreads global option
	
	if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles")
      if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "4")
         message( "MINGW make detected, adding -march=i486 flag" )
         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=i486" )
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=i486 " )
      endif()
      message( "MINGW make detected, adding -mthreads flag" )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mthreads" )
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mthreads" )
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mthreads" )
      set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -mthreads" )
	endif()
endif()

