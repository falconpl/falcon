#ifndef GTK_ACTIVATABLE_HPP
#define GTK_ACTIVATABLE_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


#if GTK_MINOR_VERSION >= 16

namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Activatable
 */
namespace Activatable {

void clsInit( Falcon::Module*, Falcon::Symbol* );

FALCON_FUNC do_set_related_action( VMARG );

FALCON_FUNC get_related_action( VMARG );

FALCON_FUNC get_use_action_appearance( VMARG );

FALCON_FUNC sync_action_properties( VMARG );

FALCON_FUNC set_related_action( VMARG );

FALCON_FUNC set_use_action_appearance( VMARG );


} // Activatable
} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 16

#endif // !GTK_ACTIVATABLE_HPP
