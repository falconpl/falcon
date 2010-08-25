#ifndef GTK_MENU_HPP
#define GTK_MENU_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Menu
 */
class Menu
    :
    public Gtk::CoreGObject
{
public:

    Menu( const Falcon::CoreClass*, const GtkMenu* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_move_scroll( VMARG );

    static void on_move_scroll( GtkMenu*, GtkScrollType, gpointer );

    static FALCON_FUNC set_screen( VMARG );

#if 0 // deprecated
    static FALCON_FUNC append( VMARG );

    static FALCON_FUNC prepend( VMARG );

    static FALCON_FUNC insert( VMARG );
#endif

    static FALCON_FUNC reorder_child( VMARG );

    static FALCON_FUNC attach( VMARG );

    static FALCON_FUNC popup( VMARG );
#if 0 // todo
    static FALCON_FUNC set_accel_group( VMARG );

    static FALCON_FUNC get_accel_group( VMARG );
#endif
    static FALCON_FUNC set_accel_path( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC get_accel_path( VMARG );
#endif

    static FALCON_FUNC set_title( VMARG );

    static FALCON_FUNC get_title( VMARG );

    static FALCON_FUNC set_monitor( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC get_monitor( VMARG );
#endif

    static FALCON_FUNC get_tearoff_state( VMARG );

#if GTK_CHECK_VERSION( 2, 18, 0 )
    static FALCON_FUNC set_reserve_toggle_size( VMARG );

    static FALCON_FUNC get_reserve_toggle_size( VMARG );
#endif

    static FALCON_FUNC popdown( VMARG );

    static FALCON_FUNC reposition( VMARG );

    static FALCON_FUNC get_active( VMARG );

    static FALCON_FUNC set_active( VMARG );

    static FALCON_FUNC set_tearoff_state( VMARG );

    static FALCON_FUNC attach_to_widget( VMARG );

    static FALCON_FUNC detach( VMARG );

    static FALCON_FUNC get_attach_widget( VMARG );

    static FALCON_FUNC get_for_attach_widget( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_MENU_HPP
