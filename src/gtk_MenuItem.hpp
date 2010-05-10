#ifndef GTK_MENUITEM_HPP
#define GTK_MENUITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::MenuItem
 */
class MenuItem
    :
    public Gtk::CoreGObject
{
public:

    MenuItem( const Falcon::CoreClass*, const GtkMenuItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_activate( VMARG );

    static void on_activate( GtkMenuItem*, gpointer );

    static FALCON_FUNC signal_activate_item( VMARG );

    static void on_activate_item( GtkMenuItem*, gpointer );

    //static FALCON_FUNC signal_toggle_size_allocate( VMARG );

    //static FALCON_FUNC signal_toggle_size_request( VMARG );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );

    static FALCON_FUNC set_right_justified( VMARG );

    static FALCON_FUNC get_right_justified( VMARG );

#if GTK_MINOR_VERSION >= 16
    static FALCON_FUNC get_label( VMARG );

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC get_use_underline( VMARG );

    static FALCON_FUNC set_use_underline( VMARG );
#endif

    static FALCON_FUNC set_submenu( VMARG );

    static FALCON_FUNC get_submenu( VMARG );

    //static FALCON_FUNC remove_submenu( VMARG );

    static FALCON_FUNC set_accel_path( VMARG );

    static FALCON_FUNC get_accel_path( VMARG );

    static FALCON_FUNC select( VMARG );

    static FALCON_FUNC deselect( VMARG );

    static FALCON_FUNC activate( VMARG );
#if 0
    static FALCON_FUNC toggle_size_request( VMARG );

    static FALCON_FUNC toggle_size_allocate( VMARG );

    static FALCON_FUNC right_justify( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_MENUITEM_HPP
