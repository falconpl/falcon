#ifndef GTK_PANED_HPP
#define GTK_PANED_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Paned
 */
class Paned
    :
    public Gtk::CoreGObject
{
public:

    Paned( const Falcon::CoreClass*, const GtkPaned* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC add1( VMARG );

    static FALCON_FUNC add2( VMARG );

    static FALCON_FUNC pack1( VMARG );

    static FALCON_FUNC pack2( VMARG );

    static FALCON_FUNC get_child1( VMARG );

    static FALCON_FUNC get_child2( VMARG );

    static FALCON_FUNC set_position( VMARG );

    static FALCON_FUNC get_position( VMARG );

#if GTK_CHECK_VERSION( 2, 20, 0 )
    static FALCON_FUNC get_handle_window( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_PANED_HPP
