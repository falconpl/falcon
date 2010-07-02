#ifndef GDK_SCREEN_HPP
#define GDK_SCREEN_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Screen
 */
class Screen
    :
    public Gtk::CoreGObject
{
public:

    Screen( const Falcon::CoreClass*, const GdkScreen* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

#if 0
    static FALCON_FUNC signal_composited_changed( VMARG );

    static FALCON_FUNC signal_monitors_changed( VMARG );

    static FALCON_FUNC signal_size_changed( VMARG );

    static FALCON_FUNC get_default( VMARG );

    static FALCON_FUNC get_default_colormap( VMARG );

    static FALCON_FUNC set_default_colormap( VMARG );

    static FALCON_FUNC get_system_colormap( VMARG );

    static FALCON_FUNC get_system_visual( VMARG );

    static FALCON_FUNC get_rgb_colormap( VMARG );

    static FALCON_FUNC get_rgb_visual( VMARG );

    static FALCON_FUNC get_rgba_colormap( VMARG );

    static FALCON_FUNC get_rgba_visual( VMARG );

    static FALCON_FUNC is_composited( VMARG );

    static FALCON_FUNC get_root_window( VMARG );

    static FALCON_FUNC get_display( VMARG );

    static FALCON_FUNC get_number( VMARG );

    static FALCON_FUNC get_width( VMARG );

    static FALCON_FUNC get_height( VMARG );

    static FALCON_FUNC get_width_mm( VMARG );

    static FALCON_FUNC get_height_mm( VMARG );

    static FALCON_FUNC list_visuals( VMARG );

    static FALCON_FUNC get_toplevel_windows( VMARG );

    static FALCON_FUNC make_display_name( VMARG );

    static FALCON_FUNC get_n_monitors( VMARG );

    static FALCON_FUNC get_primary_monitor( VMARG );

    static FALCON_FUNC get_monitor_geometry( VMARG );

    static FALCON_FUNC get_monitor_at_point( VMARG );

    static FALCON_FUNC get_monitor_at_window( VMARG );

    static FALCON_FUNC get_monitor_height_mm( VMARG );

    static FALCON_FUNC get_monitor_width_mm( VMARG );

    static FALCON_FUNC get_monitor_plug_name( VMARG );

    static FALCON_FUNC broadcast_client_message( VMARG );

    static FALCON_FUNC get_setting( VMARG );

    static FALCON_FUNC get_font_options( VMARG );

    static FALCON_FUNC set_font_options( VMARG );

    static FALCON_FUNC get_resolution( VMARG );

    static FALCON_FUNC set_resolution ( VMARG );

    static FALCON_FUNC get_active_window( VMARG );

    static FALCON_FUNC get_window_stack( VMARG );
#endif

#if 0
    gdk_spawn_on_screen
    gdk_spawn_on_screen_with_pipes
    gdk_spawn_command_line_on_screen
#endif

};


} // Gdk
} // Falcon

#endif // !GDK_SCREEN_HPP
