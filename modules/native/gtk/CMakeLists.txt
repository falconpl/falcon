##
#   Falcon GTK+ project
#
project( gtk )

cmake_minimum_required( VERSION 2.6.3 )

if(NOT FALCON_BUILD_GTK )
	message( STATUS "NOT building module: gtk")
	return()
endif()
message( STATUS "Building module: gtk" )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

##
#   Options
#
option( GTK_NO_PARAMETER_CHECK "Disable all parameters checks" OFF )
option( GTK_STRICT_PARAMETER_CHECK "Enable strictest parameters checks" OFF )
mark_as_advanced( GTK_NO_PARAMETER_CHECK )
mark_as_advanced( GTK_STRICT_PARAMETER_CHECK )

if ( GTK_STRICT_PARAMETER_CHECK )
    add_definitions( -DSTRICT_PARAMETER_CHECK )
elseif ( GTK_NO_PARAMETER_CHECK )
    add_definitions( -DNO_PARAMETER_CHECK )
endif()

##
#   Obsolete options
#
option( GTK_ARM_POKY_LINUX "Compile for ARM-Poky-Linux (eReader)" OFF )
mark_as_advanced( GTK_ARM_POKY_LINUX )

##
#   Find Falcon
#
find_package( Falcon
    HINTS "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
          "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
          "$ENV{FALCON_BIN_PATH}/cmake"
    REQUIRED )
include_directories( ${Falcon_INCLUDE_DIRS} )

##
#   Find gtk
#
if ( WIN32 )

    set( GTK_DIR ${GTK_BUNDLE_DIR} ) ## dir of the Gtk bundled stuff

    set( GTK2_ATK_INCLUDE_DIR "${GTK_DIR}/include/atk-1.0" )
    set( GTK2_ATK_LIBRARY "${GTK_DIR}/lib/atk-1.0.lib" )

    set( GTK2_CAIRO_INCLUDE_DIR "${GTK_DIR}/include/cairo" )
    set( GTK2_CAIRO_LIBRARY "${GTK_DIR}/lib/cairo.lib" )

    set( GTK2_FONTCONFIG_INCLUDE_DIR "${GTK_DIR}/include/fontconfig" )

    set( GTK2_GDKCONFIG_INCLUDE_DIR "${GTK_DIR}/lib/gtk-2.0/include" )
    set( GTK2_GDK_INCLUDE_DIR  "${GTK_DIR}/include/gtk-2.0" )
    set( GTK2_GDK_LIBRARY "${GTK_DIR}/lib/gdk-win32-2.0.lib" )

    set( GTK2_GDK_PIXBUF_INCLUDE_DIR "${GTK_DIR}/include/gdk-pixbuf-2.0" )
    set( GTK2_GDKPIXBUF_LIBRARY "${GTK_DIR}/lib/gdk_pixbuf-2.0.lib" )

    set( GTK2_GLIBCONFIG_INCLUDE_DIR "${GTK_DIR}/lib/glib-2.0/include" )
    set( GTK2_GLIB_INCLUDE_DIR "${GTK_DIR}/include/glib-2.0" )
    set( GTK2_GLIB_LIBRARY "${GTK_DIR}/lib/glib-2.0.lib" )

    set( GTK2_GOBJECT_INCLUDE_DIR "${GTK_DIR}/include/glib-2.0/gobject" )
    set( GTK2_GOBJECT_LIBRARY "${GTK_DIR}/lib/gobject-2.0.lib" )

    set( GTK2_GTHREAD_INCLUDE_DIR "${GTK_DIR}/include/glib-2.0" )
    set( GTK2_GTHREAD_LIBRARY "${GTK_DIR}/lib/gthread-2.0.lib" )

    set( GTK2_GTK_INCLUDE_DIR "${GTK_DIR}/include/gtk-2.0" )
    set( GTK2_GTK_LIBRARY "${GTK_DIR}/lib/gtk-win32-2.0.lib" )

    set( GTK2_PANGO_INCLUDE_DIR "${GTK_DIR}/include/pango-1.0" )
    set( GTK2_PANGO_LIBRARY "${GTK_DIR}/lib/pango-1.0.lib" )

    set( GTK2_GMODULE_LIBRARY "${GTK_DIR}/lib/gmodule-2.0.lib" )

    find_package( GTK2 2.12 REQUIRED )

    include_directories(
        ${GTK2_ATK_INCLUDE_DIR}
        ${GTK2_CAIRO_INCLUDE_DIR}
        ${GTK2_FONTCONFIG_INCLUDE_DIR}
        ${GTK2_GDKCONFIG_INCLUDE_DIR}
        ${GTK2_GDK_INCLUDE_DIR}
        ${GTK2_GDK_PIXBUF_INCLUDE_DIR}
        ${GTK2_GLIBCONFIG_INCLUDE_DIR}
        ${GTK2_GLIB_INCLUDE_DIR}
        ${GTK2_GOBJECT_INCLUDE_DIR}
        ${GTK2_GTK_INCLUDE_DIR}
        ${GTK2_PANGO_INCLUDE_DIR}
        )

    set( GTK2_LIBRARIES
        ${GTK2_ATK_LIBRARY}
        ${GTK2_CAIRO_LIBRARY}
        ${GTK2_GDK_LIBRARY}
        ${GTK2_GDKPIXBUF_LIBRARY}
        ${GTK2_GLIB_LIBRARY}
        ${GTK2_GOBJECT_LIBRARY}
        ${GTK2_GTHREAD_LIBRARY}
        ${GTK2_GTK_LIBRARY}
        ${GTK2_PANGO_LIBRARY}
        ${GTK2_GMODULE_LIBRARY}
        )

else() # Linux

    if ( GTK_ARM_POKY_LINUX ) # embedded (eReader)
        include( FalconGTK-ARMPokyLinux )

    else() # normal linux

        find_package( GTK2 2.12 REQUIRED )
        include_directories( ${GTK2_INCLUDE_DIRS} )

    endif()
endif()

##
#   Subdirs
#
add_subdirectory( src )

# vi: set ai et sw=4:
# kate: replace-tabs on; shift-width 4;
