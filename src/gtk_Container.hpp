#ifndef GTK_CONTAINER_HPP
#define GTK_CONTAINER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Container
 */
namespace Container {

void modInit( Falcon::Module* );

FALCON_FUNC add( VMARG );

FALCON_FUNC remove( VMARG );

//FALCON_FUNC add_with_properties( VMARG );

FALCON_FUNC get_resize_mode( VMARG );

FALCON_FUNC set_resize_mode( VMARG );

FALCON_FUNC check_resize( VMARG );

//FALCON_FUNC foreach( VMARG );

//FALCON_FUNC foreach_full( VMARG );

//FALCON_FUNC get_children( VMARG );

FALCON_FUNC set_reallocate_redraws( VMARG );

FALCON_FUNC get_focus_child( VMARG );

FALCON_FUNC set_focus_child( VMARG );

//FALCON_FUNC get_focus_vadjustment( VMARG );

//FALCON_FUNC set_focus_vadjustment( VMARG );

//FALCON_FUNC get_focus_hadjustment( VMARG );

//FALCON_FUNC set_focus_hadjustment( VMARG );

FALCON_FUNC resize_children( VMARG );

FALCON_FUNC child_type( VMARG );

//FALCON_FUNC child_get( VMARG );

//FALCON_FUNC child_set( VMARG );

//FALCON_FUNC child_get_property( VMARG );

//FALCON_FUNC child_set_property( VMARG );

//FALCON_FUNC child_get_valist( VMARG );

//FALCON_FUNC child_set_valist( VMARG );

//FALCON_FUNC forall( VMARG );

FALCON_FUNC get_border_width( VMARG );

FALCON_FUNC set_border_width( VMARG );

//FALCON_FUNC propagate_expose( VMARG );

//FALCON_FUNC get_focus_chain( VMARG );

//FALCON_FUNC unset_focus_chain( VMARG );

//FALCON_FUNC class_find_child_property( VMARG );

//FALCON_FUNC class_install_child_property( VMARG );

//FALCON_FUNC class_list_child_properties( VMARG );


} // Container
} // Gtk
} // Falcon

#endif // !GTK_CONTAINER_HPP
