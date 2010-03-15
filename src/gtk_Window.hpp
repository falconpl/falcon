#ifndef GTK_WINDOW_HPP
#define GTK_WINDOW_HPP

// namespace Gtk {
// class Widget;
// }

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Window
 */
namespace Window {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

//FALCON_FUNC signal_activate_default( VMARG );

//FALCON_FUNC signal_activate_focus( VMARG );

//FALCON_FUNC signal_frame_event( VMARG );

//FALCON_FUNC signal_keys_changed( VMARG );

//FALCON_FUNC signal_set_focus( VMARG );

FALCON_FUNC set_title( VMARG );

FALCON_FUNC set_resizable( VMARG );

FALCON_FUNC get_resizable( VMARG );

FALCON_FUNC activate_focus( VMARG );

FALCON_FUNC activate_default( VMARG );

FALCON_FUNC set_modal( VMARG );

FALCON_FUNC set_default_size( VMARG );

//FALCON_FUNC set_geometry_hints( VMARG );

FALCON_FUNC set_gravity( VMARG );

FALCON_FUNC get_gravity( VMARG );

FALCON_FUNC set_position( VMARG );

FALCON_FUNC set_transient_for( VMARG );

FALCON_FUNC set_destroy_with_parent( VMARG );

//FALCON_FUNC set_screen( VMARG );

//FALCON_FUNC get_screen( VMARG );

FALCON_FUNC is_active( VMARG );

FALCON_FUNC has_toplevel_focus( VMARG );

//FALCON_FUNC list_toplevels( VMARG );

FALCON_FUNC add_mnemonic( VMARG );

FALCON_FUNC remove_mnemonic( VMARG );

FALCON_FUNC mnemonic_activate( VMARG );

//FALCON_FUNC activate_key( VMARG );

//FALCON_FUNC propagate_key_event( VMARG );

FALCON_FUNC get_focus( VMARG );

FALCON_FUNC set_focus( VMARG );

FALCON_FUNC get_default_widget( VMARG );

FALCON_FUNC set_default( VMARG );

FALCON_FUNC present( VMARG );

FALCON_FUNC present_with_time( VMARG );

FALCON_FUNC iconify( VMARG );

FALCON_FUNC deiconify( VMARG );

FALCON_FUNC stick( VMARG );

FALCON_FUNC unstick( VMARG );

FALCON_FUNC maximize( VMARG );

FALCON_FUNC unmaximize( VMARG );

FALCON_FUNC fullscreen( VMARG );

FALCON_FUNC unfullscreen( VMARG );

FALCON_FUNC set_keep_above( VMARG );

FALCON_FUNC set_keep_below( VMARG );

FALCON_FUNC begin_resize_drag( VMARG );

FALCON_FUNC begin_move_drag( VMARG );

FALCON_FUNC set_decorated( VMARG );

FALCON_FUNC set_deletable( VMARG );

FALCON_FUNC set_frame_dimensions( VMARG );

FALCON_FUNC set_has_frame( VMARG );

//FALCON_FUNC set_mnemonic_modifier( VMARG );

//FALCON_FUNC set_type_hint( VMARG );

//FALCON_FUNC set_skip_taskbar_hint( VMARG );

//FALCON_FUNC set_skip_pager_hint( VMARG );

//FALCON_FUNC set_urgency_hint( VMARG );

//FALCON_FUNC set_accept_focus( VMARG );

//FALCON_FUNC set_focus_on_map( VMARG );

//FALCON_FUNC set_startup_id( VMARG );

//FALCON_FUNC set_role( VMARG );

//FALCON_FUNC get_decorated( VMARG );

//FALCON_FUNC get_deletable( VMARG );

//FALCON_FUNC get_default_icon_list( VMARG );

//FALCON_FUNC get_default_icon_name( VMARG );

//FALCON_FUNC get_default_size( VMARG );

//FALCON_FUNC get_destroy_with_parent( VMARG );

//FALCON_FUNC get_frame_dimensions( VMARG );

//FALCON_FUNC get_has_frame( VMARG );

//FALCON_FUNC get_icon( VMARG );

//FALCON_FUNC get_icon_list( VMARG );

//FALCON_FUNC get_icon_name( VMARG );

//FALCON_FUNC get_mnemonic_modifier( VMARG );

//FALCON_FUNC get_modal( VMARG );

//FALCON_FUNC get_position( VMARG );

//FALCON_FUNC get_role( VMARG );

//FALCON_FUNC get_size( VMARG );

FALCON_FUNC get_title( VMARG );

//FALCON_FUNC get_transient_for( VMARG );

//FALCON_FUNC get_type_hint( VMARG );

//FALCON_FUNC get_skip_taskbar_hint( VMARG );

//FALCON_FUNC get_skip_pager_hint( VMARG );

//FALCON_FUNC get_urgency_hint( VMARG );

//FALCON_FUNC get_accept_focus( VMARG );

//FALCON_FUNC get_focus_on_map( VMARG );

//FALCON_FUNC get_group( VMARG );

//FALCON_FUNC get_window_type( VMARG );

//FALCON_FUNC move( VMARG );

//FALCON_FUNC parse_geometry( VMARG );

//FALCON_FUNC reshow_with_initial_size( VMARG );

//FALCON_FUNC resize( VMARG );

//FALCON_FUNC set_default_icon_list( VMARG );

//FALCON_FUNC set_default_icon( VMARG );

//FALCON_FUNC set_default_icon_from_file( VMARG );

//FALCON_FUNC set_default_icon_name( VMARG );

//FALCON_FUNC set_icon( VMARG );

//FALCON_FUNC set_icon_list( VMARG );

//FALCON_FUNC set_icon_from_file( VMARG );

//FALCON_FUNC set_icon_name( VMARG );

//FALCON_FUNC set_auto_startup_notification( VMARG );

//FALCON_FUNC get_opacity( VMARG );

//FALCON_FUNC set_opacity( VMARG );

//FALCON_FUNC get_mnemonics_visible( VMARG );

//FALCON_FUNC set_mnemonics_visible( VMARG );


} // Window
} // Gtk
} // Falcon

#endif // !GTK_WINDOW_HPP
