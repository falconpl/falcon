##
#   Falcon GTK+ project
#
project( gtk )

cmake_minimum_required( VERSION 2.8 )

##
#   Find Falcon
#
find_package( Falcon
    HINTS "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
          "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
          "$ENV{FALCON_BIN_PATH}/cmake"
    REQUIRED )
include_directories( ${Falcon_INCLUDE_DIRS} )
link_directories( ${FALCON_LIB_PATH} )

##
#   Find gtk
#
IF(WIN32)
    SET( GTK2_GTK_LIBRARY "$ENV{GTK_ROOT}/lib/gtk-win32-2.0.lib" )
    SET( GTK2_GTK_INCLUDE_DIR "$ENV{GTK_ROOT}/include/gtk-2.0" )
    SET( GTK2_GLIB_INCLUDE_DIR "$ENV{GTK_ROOT}/include/glib-2.0" )
    SET( GTK2_GLIBCONFIG_INCLUDE_DIR "$ENV{GTK_ROOT}/lib/glib-2.0/include" )
    SET( GTK2_GLIB_LIBRARY "$ENV{GTK_ROOT}/lib/glib-2.0.lib" )
    SET( GTK2_GDK_INCLUDE_DIR  "$ENV{GTK_ROOT}/include/gtk-2.0" )
    SET( GTK2_GDKCONFIG_INCLUDE_DIR "$ENV{GTK_ROOT}/lib/gtk-2.0/include" )
    SET( GTK2_GDK_LIBRARY "$ENV{GTK_ROOT}/lib/gdk-win32-2.0.lib" )
    SET( GTK2_ATK_LIBRARY "$ENV{GTK_ROOT}/lib/atk-1.0.lib" )
    SET( GTK2_CAIRO_LIBRARY "$ENV{GTK_ROOT}/lib/cairo.lib" )
    SET( GTK2_PANGO_LIBRARY "$ENV{GTK_ROOT}/lib/pango-1.0.lib" )

    find_package( GTK2 REQUIRED)	

    SET( GTK2_LIBRARIES "${GTK2_LIBRARIES};$ENV{GTK_ROOT}/lib/gobject-2.0.lib" )
ELSE(WIN32)
    find_package( GTK2 REQUIRED )
    include_directories( ${GTK2_INCLUDE_DIRS} )
ENDIF(WIN32)

##
#   Subdirs
#
add_subdirectory( src )
