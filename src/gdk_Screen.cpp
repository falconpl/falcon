/**
 *  \file gdk_Screen.cpp
 */

#include "gdk_Screen.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Screen::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Screen = mod->addClass( "GdkScreen", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Screen->getClassDef()->addInheritance( in );

    c_Screen->setWKS( true );
    c_Screen->getClassDef()->factory( &Screen::factory );

    Gtk::MethodTab methods[] =
    {
#if 0
    { "get_default",        &Screen:: },
    { "get_default_colormap",        &Screen:: },
    { "set_default_colormap",        &Screen:: },
    { "get_system_colormap",        &Screen:: },
    { "get_system_visual",        &Screen:: },
    { "get_rgb_colormap",        &Screen:: },
    { "get_rgb_visual",        &Screen:: },
    { "get_rgba_colormap",        &Screen:: },
    { "get_rgba_visual",        &Screen:: },
    { "is_composited",        &Screen:: },
    { "get_root_window",        &Screen:: },
    { "get_display",        &Screen:: },
    { "get_number",        &Screen:: },
    { "get_width",        &Screen:: },
    { "get_height",        &Screen:: },
    { "get_width_mm",        &Screen:: },
    { "get_height_mm",        &Screen:: },
    { "list_visuals",        &Screen:: },
    { "get_toplevel_windows",        &Screen:: },
    { "make_display_name",        &Screen:: },
    { "get_n_monitors",        &Screen:: },
    { "get_primary_monitor",        &Screen:: },
    { "get_monitor_geometry",        &Screen:: },
    { "get_monitor_at_point",        &Screen:: },
    { "get_monitor_at_window",        &Screen:: },
    { "get_monitor_height_mm",        &Screen:: },
    { "get_monitor_width_mm",        &Screen:: },
    { "get_monitor_plug_name",        &Screen:: },
    { "broadcast_client_message",        &Screen:: },
    { "get_setting",        &Screen:: },
    { "get_font_options",        &Screen:: },
    { "set_font_options",        &Screen:: },
    { "get_resolution",        &Screen:: },
    { "set_resolution ",        &Screen:: },
    { "get_active_window",        &Screen:: },
    { "get_window_stack",        &Screen:: },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Screen, meth->name, meth->cb );
}


Screen::Screen( const Falcon::CoreClass* gen, const GdkScreen* scrn )
    :
    Gtk::CoreGObject( gen, (GObject*) scrn )
{}


Falcon::CoreObject* Screen::factory( const Falcon::CoreClass* gen, void* scrn, bool )
{
    return new Screen( gen, (GdkScreen*) scrn );
}


/*#
    @class GdkScreen
    @brief Object representing a physical screen

    GdkScreen objects are the GDK representation of a physical screen. It is used
    throughout GDK and GTK+ to specify which screen the top level windows are to
    be displayed on. It is also used to query the screen specification and default
    settings such as the default colormap (gdk_screen_get_default_colormap()),
    the screen width (gdk_screen_get_width()), etc.

    Note that a screen may consist of multiple monitors which are merged to form a large screen area.
 */

#if 0
FALCON_FUNC Screen::signal_composited_changed( VMARG );

FALCON_FUNC Screen::signal_monitors_changed( VMARG );

FALCON_FUNC Screen::signal_size_changed( VMARG );


/*#
    @method get_default
    @brief Gets the default screen for the default display.
    @return a GdkScreen, or nil if there is no default display.
 */
FALCON_FUNC Screen::get_default( VMARG )
{
    NO_ARGS
    GdkScreen* scrn = gdk_screen_get_default();
    if ( scrn )
        vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(), scrn ) );
    else
        vm->retnil();
}



FALCON_FUNC Screen::get_default_colormap( VMARG );
FALCON_FUNC Screen::set_default_colormap( VMARG );
FALCON_FUNC Screen::get_system_colormap( VMARG );
FALCON_FUNC Screen::get_system_visual( VMARG );
FALCON_FUNC Screen::get_rgb_colormap( VMARG );
FALCON_FUNC Screen::get_rgb_visual( VMARG );
FALCON_FUNC Screen::get_rgba_colormap( VMARG );
FALCON_FUNC Screen::get_rgba_visual( VMARG );
FALCON_FUNC Screen::is_composited( VMARG );
FALCON_FUNC Screen::get_root_window( VMARG );
FALCON_FUNC Screen::get_display( VMARG );
FALCON_FUNC Screen::get_number( VMARG );
FALCON_FUNC Screen::get_width( VMARG );
FALCON_FUNC Screen::get_height( VMARG );
FALCON_FUNC Screen::get_width_mm( VMARG );
FALCON_FUNC Screen::get_height_mm( VMARG );
FALCON_FUNC Screen::list_visuals( VMARG );
FALCON_FUNC Screen::get_toplevel_windows( VMARG );
FALCON_FUNC Screen::make_display_name( VMARG );
FALCON_FUNC Screen::get_n_monitors( VMARG );
FALCON_FUNC Screen::get_primary_monitor( VMARG );
FALCON_FUNC Screen::get_monitor_geometry( VMARG );
FALCON_FUNC Screen::get_monitor_at_point( VMARG );
FALCON_FUNC Screen::get_monitor_at_window( VMARG );
FALCON_FUNC Screen::get_monitor_height_mm( VMARG );
FALCON_FUNC Screen::get_monitor_width_mm( VMARG );
FALCON_FUNC Screen::get_monitor_plug_name( VMARG );
FALCON_FUNC Screen::broadcast_client_message( VMARG );
FALCON_FUNC Screen::get_setting( VMARG );
FALCON_FUNC Screen::get_font_options( VMARG );
FALCON_FUNC Screen::set_font_options( VMARG );
FALCON_FUNC Screen::get_resolution( VMARG );
FALCON_FUNC Screen::set_resolution ( VMARG );
FALCON_FUNC Screen::get_active_window( VMARG );
FALCON_FUNC Screen::get_window_stack( VMARG );
#endif


} // Gdk
} // Falcon
