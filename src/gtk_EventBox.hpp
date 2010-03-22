#ifndef GTK_EVENTBOX_HPP
#define GTK_EVENTBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::EventBox
 */
namespace EventBox {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC set_above_child( VMARG );

FALCON_FUNC get_above_child( VMARG );

FALCON_FUNC set_visible_window( VMARG );

FALCON_FUNC get_visible_window( VMARG );

} // EventBox
} // Gtk
} // Falcon

#endif // !GTK_EVENTBOX_HPP
