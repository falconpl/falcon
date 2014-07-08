#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/rpath.cmake
#
#   RPATH Management.
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

if(NOT DISABLE_RPATH)
  # Always find libfalcon_engine.so in build and install tree, without LD_LIBRARY_PATH.
  set(CMAKE_SKIP_BUILD_RPATH  false)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH false)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${FALCON_LIB_DIR}")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH false)

  # Apple equivalent to RPATH is called `install_name'
  set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${FALCON_LIB_DIR}")

else()
  set(CMAKE_SKIP_RPATH on)
endif()
