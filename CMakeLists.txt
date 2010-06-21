##
#   Falcon GTK+ project
#
project( gtk )

cmake_minimum_required( VERSION 2.8 )

##
#   Options
#
option( ARM_POKY_LINUX "Compile for ARM-Poky-Linux (eReader)" OFF )
option( NO_PARAMETER_CHECK "Disable all parameters checks" OFF )
option( STRICT_PARAMETER_CHECK "Enable strictest parameters checks" OFF )

if ( STRICT_PARAMETER_CHECK )
    add_definitions( -DSTRICT_PARAMETER_CHECK )
elseif ( NO_PARAMETER_CHECK )
    add_definitions( -DNO_PARAMETER_CHECK )
endif()

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
if ( WIN32 ) # Windoze, using env var GTK_ROOT

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

    find_package( GTK2 2.12 REQUIRED )

    set( GTK2_LIBRARIES "${GTK2_LIBRARIES};$ENV{GTK_ROOT}/lib/gobject-2.0.lib" )

else() # Linux

    if ( ARM_POKY_LINUX ) # embedded (eReader)

        message( STATUS "-- Compiling for ARM-Poky-Linux..." )

        set( POKY_ROOT /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi )

        include_directories( BEFORE
            ${POKY_ROOT}/usr/include/gtk-2.0
            ${POKY_ROOT}/usr/lib/gtk-2.0/include
            ${POKY_ROOT}/usr/include/glib-2.0
            ${POKY_ROOT}/usr/lib/glib-2.0/include
            ${POKY_ROOT}/usr/include/pango-1.0
            ${POKY_ROOT}/usr/include/cairo
            ${POKY_ROOT}/usr/include/atk-1.0 )

        set( GTK2_LIBRARIES
            ${POKY_ROOT}/usr/lib/libglib-2.0.so
            ${POKY_ROOT}/usr/lib/libgobject-2.0.so
            ${POKY_ROOT}/usr/lib/libgdk-x11-2.0.so
            ${POKY_ROOT}/usr/lib/libgtk-x11-2.0.so
            ${POKY_ROOT}/usr/lib/libcairo.so
            ${POKY_ROOT}/usr/lib/libpango-1.0.so
            ${POKY_ROOT}/usr/lib/libatk-1.0.so )

    else() # normal linux

        find_package( GTK2 2.12 REQUIRED )
        include_directories( ${GTK2_INCLUDE_DIRS} )

    endif()
endif()

##
#   Subdirs
#
add_subdirectory( src )
