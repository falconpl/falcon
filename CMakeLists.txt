##
#   Falcon GTK+ project
#
project( gtk )

cmake_minimum_required( VERSION 2.8 )

##
#   Find Falcon
#
find_package( falcon
    HINTS "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
          "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
          "$ENV{FALCON_BIN_PATH}/cmake"
    REQUIRED )
include_directories( ${falcon_INCLUDE_DIRS} )
link_directories( ${FALCON_LIB_PATH} )

##
#   Find gtk
#
find_package( GTK2 REQUIRED )
include_directories( ${GTK2_INCLUDE_DIRS} )

##
#   Subdirs
#
include( "falmod.cmake" )
add_subdirectory( src )
