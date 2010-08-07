#ifndef GTK_CONTAINER_HPP
#define GTK_CONTAINER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Container
 */
class Container
    :
    public Gtk::CoreGObject
{
public:

    Container( const Falcon::CoreClass*, const GtkContainer* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC add( VMARG );

    static FALCON_FUNC remove( VMARG );

    //static FALCON_FUNC add_with_properties( VMARG );

    static FALCON_FUNC get_resize_mode( VMARG );

    static FALCON_FUNC set_resize_mode( VMARG );

    static FALCON_FUNC check_resize( VMARG );

    //static FALCON_FUNC foreach( VMARG );

    //static FALCON_FUNC foreach_full( VMARG );

    //static FALCON_FUNC get_children( VMARG );

    static FALCON_FUNC set_reallocate_redraws( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC get_focus_child( VMARG );
#endif

    static FALCON_FUNC set_focus_child( VMARG );

    //static FALCON_FUNC get_focus_vadjustment( VMARG );

    //static FALCON_FUNC set_focus_vadjustment( VMARG );

    //static FALCON_FUNC get_focus_hadjustment( VMARG );

    //static FALCON_FUNC set_focus_hadjustment( VMARG );

    static FALCON_FUNC resize_children( VMARG );

    static FALCON_FUNC child_type( VMARG );

    //static FALCON_FUNC child_get( VMARG );

    //static FALCON_FUNC child_set( VMARG );

    //static FALCON_FUNC child_get_property( VMARG );

    //static FALCON_FUNC child_set_property( VMARG );

    //static FALCON_FUNC child_get_valist( VMARG );

    //static FALCON_FUNC child_set_valist( VMARG );

    //static FALCON_FUNC forall( VMARG );

    static FALCON_FUNC get_border_width( VMARG );

    static FALCON_FUNC set_border_width( VMARG );

    //static FALCON_FUNC propagate_expose( VMARG );

    //static FALCON_FUNC get_focus_chain( VMARG );

    //static FALCON_FUNC unset_focus_chain( VMARG );

    //static FALCON_FUNC class_find_child_property( VMARG );

    //static FALCON_FUNC class_install_child_property( VMARG );

    //static FALCON_FUNC class_list_child_properties( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CONTAINER_HPP
