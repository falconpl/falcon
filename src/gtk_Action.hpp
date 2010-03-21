#ifndef GTK_ACTION_HPP
#define GTK_ACTION_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Action
 */
namespace Action {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC signal_activate( VMARG );

void on_activate( GtkAction*, gpointer );

//FALCON_FUNC get_name( VMARG );

//FALCON_FUNC is_sensitive( VMARG );

//FALCON_FUNC get_sensitive( VMARG );

//FALCON_FUNC set_sensitive( VMARG );

//FALCON_FUNC is_visible( VMARG );

//FALCON_FUNC get_visible( VMARG );

//FALCON_FUNC set_visible( VMARG );

//FALCON_FUNC activate( VMARG );

//FALCON_FUNC create_icon( VMARG );

//FALCON_FUNC create_menu_item( VMARG );

//FALCON_FUNC create_tool_item( VMARG );

//FALCON_FUNC create_menu( VMARG );

//FALCON_FUNC connect_proxy( VMARG );

//FALCON_FUNC disconnect_proxy( VMARG );

//FALCON_FUNC get_proxies( VMARG );

//FALCON_FUNC connect_accelerator( VMARG );

//FALCON_FUNC disconnect_accelerator( VMARG );

//FALCON_FUNC block_activate( VMARG );

//FALCON_FUNC unblock_activate( VMARG );

//FALCON_FUNC block_activate_from( VMARG );

//FALCON_FUNC unblock_activate_from( VMARG );

//FALCON_FUNC get_always_show_image( VMARG );

//FALCON_FUNC set_always_show_image( VMARG );

//FALCON_FUNC get_accel_path( VMARG );

//FALCON_FUNC set_accel_path( VMARG );

//FALCON_FUNC get_accel_closure( VMARG );

//FALCON_FUNC set_accel_group( VMARG );

//FALCON_FUNC set_label( VMARG );

//FALCON_FUNC get_label( VMARG );

//FALCON_FUNC set_short_label( VMARG );

//FALCON_FUNC get_short_label( VMARG );

//FALCON_FUNC set_tooltip( VMARG );

//FALCON_FUNC get_tooltip( VMARG );

//FALCON_FUNC set_stock_id( VMARG );

//FALCON_FUNC get_stock_id( VMARG );

//FALCON_FUNC set_gicon( VMARG );

//FALCON_FUNC get_gicon( VMARG );

//FALCON_FUNC set_icon_name( VMARG );

//FALCON_FUNC get_icon_name( VMARG );

//FALCON_FUNC set_visible_horizontal( VMARG );

//FALCON_FUNC get_visible_horizontal( VMARG );

//FALCON_FUNC set_visible_vertical( VMARG );

//FALCON_FUNC get_visible_vertical( VMARG );

//FALCON_FUNC set_is_important( VMARG );

//FALCON_FUNC get_is_important( VMARG );


} // Action
} // Gtk
} // Falcon

#endif // !GTK_ACTION_HPP
