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
if (WIN32)
    set( GTK2_GTK_LIBRARY "$ENV{GTK_ROOT}/lib/gtk-win32-2.0.lib" )
    set( GTK2_GTK_INCLUDE_DIR "$ENV{GTK_ROOT}/include/gtk-2.0" )
    set( GTK2_GLIB_INCLUDE_DIR "$ENV{GTK_ROOT}/include/glib-2.0" )
    set( GTK2_GLIBCONFIG_INCLUDE_DIR "$ENV{GTK_ROOT}/lib/glib-2.0/include" )
    set( GTK2_GLIB_LIBRARY "$ENV{GTK_ROOT}/lib/glib-2.0.lib" )
    set( GTK2_GDK_INCLUDE_DIR  "$ENV{GTK_ROOT}/include/gtk-2.0" )
    set( GTK2_GDKCONFIG_INCLUDE_DIR "$ENV{GTK_ROOT}/lib/gtk-2.0/include" )
    set( GTK2_GDK_LIBRARY "$ENV{GTK_ROOT}/lib/gdk-win32-2.0.lib" )
    set( GTK2_ATK_LIBRARY "$ENV{GTK_ROOT}/lib/atk-1.0.lib" )
    set( GTK2_CAIRO_LIBRARY "$ENV{GTK_ROOT}/lib/cairo.lib" )
    set( GTK2_PANGO_LIBRARY "$ENV{GTK_ROOT}/lib/pango-1.0.lib" )

    find_package( GTK2 REQUIRED )

    set( GTK2_LIBRARIES "${GTK2_LIBRARIES};$ENV{GTK_ROOT}/lib/gobject-2.0.lib" )
else (WIN32)
    find_package( GTK2 REQUIRED )
    include_directories( ${GTK2_INCLUDE_DIRS} )

    # eReader includes
    #include_directories( BEFORE
    #    /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi/usr/include/gtk-2.0
    #    /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi/usr/lib/gtk-2.0/include
    #    /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi/usr/include/glib-2.0
    #    /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi/usr/lib/glib-2.0/include
    #    /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi/usr/include/pango-1.0
    #    /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi/usr/include/cairo
    #    /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi/usr/include/atk-1.0
    #    )
    #
    # eReader libs
    ## todo?

endif (WIN32)

##
#   Subdirs
#
add_subdirectory( src )
