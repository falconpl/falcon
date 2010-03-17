#ifndef GTK_WINDOW_HPP
#define GTK_WINDOW_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Window
 */
class Window
    :
    public Gtk::CoreGObject
{
public:

    Window( const Falcon::CoreClass*, const GtkWindow* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC signal_activate_default( VMARG );

    //static FALCON_FUNC signal_activate_focus( VMARG );

    //static FALCON_FUNC signal_frame_event( VMARG );

    //static FALCON_FUNC signal_keys_changed( VMARG );

    //static FALCON_FUNC signal_set_focus( VMARG );

    static FALCON_FUNC set_title( VMARG );

    static FALCON_FUNC set_resizable( VMARG );

    static FALCON_FUNC get_resizable( VMARG );

    static FALCON_FUNC activate_focus( VMARG );

    static FALCON_FUNC activate_default( VMARG );

    static FALCON_FUNC set_modal( VMARG );

    static FALCON_FUNC set_default_size( VMARG );

    //static FALCON_FUNC set_geometry_hints( VMARG );

    static FALCON_FUNC set_gravity( VMARG );

    static FALCON_FUNC get_gravity( VMARG );

    static FALCON_FUNC set_position( VMARG );

    static FALCON_FUNC set_transient_for( VMARG );

    static FALCON_FUNC set_destroy_with_parent( VMARG );

    //static FALCON_FUNC set_screen( VMARG );

    //static FALCON_FUNC get_screen( VMARG );

    static FALCON_FUNC is_active( VMARG );

    static FALCON_FUNC has_toplevel_focus( VMARG );

    //static FALCON_FUNC list_toplevels( VMARG );

    static FALCON_FUNC add_mnemonic( VMARG );

    static FALCON_FUNC remove_mnemonic( VMARG );

    static FALCON_FUNC mnemonic_activate( VMARG );

    //static FALCON_FUNC activate_key( VMARG );

    //static FALCON_FUNC propagate_key_event( VMARG );

    static FALCON_FUNC get_focus( VMARG );

    static FALCON_FUNC set_focus( VMARG );

    static FALCON_FUNC get_default_widget( VMARG );

    static FALCON_FUNC set_default( VMARG );

    static FALCON_FUNC present( VMARG );

    static FALCON_FUNC present_with_time( VMARG );

    static FALCON_FUNC iconify( VMARG );

    static FALCON_FUNC deiconify( VMARG );

    static FALCON_FUNC stick( VMARG );

    static FALCON_FUNC unstick( VMARG );

    static FALCON_FUNC maximize( VMARG );

    static FALCON_FUNC unmaximize( VMARG );

    static FALCON_FUNC fullscreen( VMARG );

    static FALCON_FUNC unfullscreen( VMARG );

    static FALCON_FUNC set_keep_above( VMARG );

    static FALCON_FUNC set_keep_below( VMARG );

    static FALCON_FUNC begin_resize_drag( VMARG );

    static FALCON_FUNC begin_move_drag( VMARG );

    static FALCON_FUNC set_decorated( VMARG );

    static FALCON_FUNC set_deletable( VMARG );

    static FALCON_FUNC set_frame_dimensions( VMARG );

    static FALCON_FUNC set_has_frame( VMARG );

    static FALCON_FUNC set_mnemonic_modifier( VMARG );

    static FALCON_FUNC set_type_hint( VMARG );

    static FALCON_FUNC set_skip_taskbar_hint( VMARG );

    static FALCON_FUNC set_skip_pager_hint( VMARG );

    static FALCON_FUNC set_urgency_hint( VMARG );

    static FALCON_FUNC set_accept_focus( VMARG );

    static FALCON_FUNC set_focus_on_map( VMARG );

    static FALCON_FUNC set_startup_id( VMARG );

    static FALCON_FUNC set_role( VMARG );

    static FALCON_FUNC get_decorated( VMARG );

    static FALCON_FUNC get_deletable( VMARG );

    //static FALCON_FUNC get_default_icon_list( VMARG );

#if GTK_MINOR_VERSION >= 16
    static FALCON_FUNC get_default_icon_name( VMARG );
#endif

    static FALCON_FUNC get_default_size( VMARG );

    static FALCON_FUNC get_destroy_with_parent( VMARG );

    static FALCON_FUNC get_frame_dimensions( VMARG );

    static FALCON_FUNC get_has_frame( VMARG );

    //static FALCON_FUNC get_icon( VMARG );

    //static FALCON_FUNC get_icon_list( VMARG );

    static FALCON_FUNC get_icon_name( VMARG );

    static FALCON_FUNC get_mnemonic_modifier( VMARG );

    static FALCON_FUNC get_modal( VMARG );

    static FALCON_FUNC get_position( VMARG );

    static FALCON_FUNC get_role( VMARG );

    static FALCON_FUNC get_size( VMARG );

    static FALCON_FUNC get_title( VMARG );

    static FALCON_FUNC get_transient_for( VMARG );

    static FALCON_FUNC get_type_hint( VMARG );

    static FALCON_FUNC get_skip_taskbar_hint( VMARG );

    static FALCON_FUNC get_skip_pager_hint( VMARG );

    static FALCON_FUNC get_urgency_hint( VMARG );

    static FALCON_FUNC get_accept_focus( VMARG );

    static FALCON_FUNC get_focus_on_map( VMARG );

    //static FALCON_FUNC get_group( VMARG );

#if GTK_MINOR_VERSION >= 20
    static FALCON_FUNC get_window_type( VMARG );
#endif

    static FALCON_FUNC move( VMARG );

    static FALCON_FUNC parse_geometry( VMARG );

    static FALCON_FUNC reshow_with_initial_size( VMARG );

    static FALCON_FUNC resize( VMARG );

    //static FALCON_FUNC set_default_icon_list( VMARG );

    //static FALCON_FUNC set_default_icon( VMARG );

    //static FALCON_FUNC set_default_icon_from_file( VMARG );

    static FALCON_FUNC set_default_icon_name( VMARG );

    //static FALCON_FUNC set_icon( VMARG );

    //static FALCON_FUNC set_icon_list( VMARG );

    //static FALCON_FUNC set_icon_from_file( VMARG );

    static FALCON_FUNC set_icon_name( VMARG );

    static FALCON_FUNC set_auto_startup_notification( VMARG );

    static FALCON_FUNC get_opacity( VMARG );

    static FALCON_FUNC set_opacity( VMARG );

    //static FALCON_FUNC get_mnemonics_visible( VMARG );

    //static FALCON_FUNC set_mnemonics_visible( VMARG );


};


} // Gtk
} // Falcon

#endif // !GTK_WINDOW_HPP
