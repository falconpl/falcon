#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/sys_setup-sunos.cmake
#
#   Sun OS specific build system setup
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

if(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS")
   add_definitions( -D_POSIX_PTHREAD_SEMANTICS )
endif()

