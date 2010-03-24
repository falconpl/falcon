#ifndef GTK_PANED_HPP
#define GTK_PANED_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Paned
 */
namespace Paned {

void modInit( Falcon::Module* );

FALCON_FUNC add1( VMARG );

FALCON_FUNC add2( VMARG );

FALCON_FUNC pack1( VMARG );

FALCON_FUNC pack2( VMARG );

FALCON_FUNC get_child1( VMARG );

FALCON_FUNC get_child2( VMARG );

FALCON_FUNC set_position( VMARG );

FALCON_FUNC get_position( VMARG );

#if GTK_MINOR_VERSION >= 20
FALCON_FUNC get_handle_window( VMARG );
#endif


} // Paned
} // Gtk
} // Falcon

#endif // !GTK_PANED_HPP
