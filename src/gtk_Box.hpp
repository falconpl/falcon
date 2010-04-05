#ifndef GTK_BOX_HPP
#define GTK_BOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Box
 */
class Box
    :
    public Gtk::CoreGObject
{
public:

    Box( const Falcon::CoreClass*, const GtkBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC pack_start( VMARG );

    static FALCON_FUNC pack_end( VMARG );

    static FALCON_FUNC pack_start_defaults( VMARG );

    static FALCON_FUNC pack_end_defaults( VMARG );

    static FALCON_FUNC get_homogeneous( VMARG );

    static FALCON_FUNC set_homogeneous( VMARG );

    static FALCON_FUNC get_spacing( VMARG );

    static FALCON_FUNC reorder_child( VMARG );

    //static FALCON_FUNC query_child_packing( VMARG );

    //static FALCON_FUNC set_child_packing( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_BOX_HPP
