#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /modules/native/CMakeLists.txt
#
#   CMake build system list for native (C/C++/asm) modules
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

#####################################################
# Internal language modules
#

add_subdirectory(feathers)

#####################################################
# Other canonical native modules
#
# Note: Some module are disabled in ALPHA

# add_subdirectory(conio)
#add_subdirectory(curl)
add_subdirectory(dbi)
# add_subdirectory(dbus)
add_subdirectory(dmtx)
# add_subdirectory(dynlib)
add_subdirectory(gd2)
# add_subdirectory(gtk)
# add_subdirectory(hpdf)
# add_subdirectory(mongodb)
# add_subdirectory(MP)
# add_subdirectory(sdl)
add_subdirectory(vfs_providers)

