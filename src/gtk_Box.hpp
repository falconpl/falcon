#ifndef GTK_BOX_HPP
#define GTK_BOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Box
 */
namespace Box {

void modInit( Falcon::Module* );

FALCON_FUNC pack_start( VMARG );

FALCON_FUNC pack_end( VMARG );

FALCON_FUNC pack_start_defaults( VMARG );

FALCON_FUNC pack_end_defaults( VMARG );

FALCON_FUNC get_homogeneous( VMARG );

FALCON_FUNC set_homogeneous( VMARG );

FALCON_FUNC get_spacing( VMARG );

FALCON_FUNC reorder_child( VMARG );

//FALCON_FUNC query_child_packing( VMARG );

//FALCON_FUNC set_child_packing( VMARG );


} // Box
} // Gtk
} // Falcon

#endif // !GTK_BOX_HPP
