#ifndef GTK_ACTION_HPP
#define GTK_ACTION_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Action
 */
class Action
    :
    public Gtk::CoreGObject
{
public:

    Action( const Falcon::CoreClass*, const GtkAction* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_activate( VMARG );

    static void on_activate( GtkAction*, gpointer );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC is_sensitive( VMARG );

    static FALCON_FUNC get_sensitive( VMARG );

    static FALCON_FUNC set_sensitive( VMARG );

    static FALCON_FUNC is_visible( VMARG );

    static FALCON_FUNC get_visible( VMARG );

    static FALCON_FUNC set_visible( VMARG );

    static FALCON_FUNC activate( VMARG );

    static FALCON_FUNC create_icon( VMARG );

    static FALCON_FUNC create_menu_item( VMARG );

    static FALCON_FUNC create_tool_item( VMARG );

    static FALCON_FUNC create_menu( VMARG );

#if 0 // deprecated
    static FALCON_FUNC connect_proxy( VMARG );
    static FALCON_FUNC disconnect_proxy( VMARG );
#endif

    static FALCON_FUNC get_proxies( VMARG );

    static FALCON_FUNC connect_accelerator( VMARG );

    static FALCON_FUNC disconnect_accelerator( VMARG );

#if GTK_CHECK_VERSION( 2, 16, 0 )
    static FALCON_FUNC block_activate( VMARG );

    static FALCON_FUNC unblock_activate( VMARG );
#endif

#if 0 // deprecated
    static FALCON_FUNC block_activate_from( VMARG );
    static FALCON_FUNC unblock_activate_from( VMARG );
#endif

#if GTK_CHECK_VERSION( 2, 20, 0 )
    static FALCON_FUNC get_always_show_image( VMARG );

    static FALCON_FUNC set_always_show_image( VMARG );
#endif

    static FALCON_FUNC get_accel_path( VMARG );

    static FALCON_FUNC set_accel_path( VMARG );

    //static FALCON_FUNC get_accel_closure( VMARG );

    static FALCON_FUNC set_accel_group( VMARG );

#if GTK_CHECK_VERSION( 2, 16, 0 )

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC get_label( VMARG );

    static FALCON_FUNC set_short_label( VMARG );

    static FALCON_FUNC get_short_label( VMARG );

    static FALCON_FUNC set_tooltip( VMARG );

    static FALCON_FUNC get_tooltip( VMARG );

    static FALCON_FUNC set_stock_id( VMARG );

    static FALCON_FUNC get_stock_id( VMARG );

    //static FALCON_FUNC set_gicon( VMARG );

    //static FALCON_FUNC get_gicon( VMARG );

    static FALCON_FUNC set_icon_name( VMARG );

    static FALCON_FUNC get_icon_name( VMARG );

    static FALCON_FUNC set_visible_horizontal( VMARG );

    static FALCON_FUNC get_visible_horizontal( VMARG );

    static FALCON_FUNC set_visible_vertical( VMARG );

    static FALCON_FUNC get_visible_vertical( VMARG );

    static FALCON_FUNC set_is_important( VMARG );

    static FALCON_FUNC get_is_important( VMARG );

#endif // GTK_CHECK_VERSION( 2, 16, 0 )

};


} // Gtk
} // Falcon

#endif // !GTK_ACTION_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
