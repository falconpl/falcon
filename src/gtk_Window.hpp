#ifndef GTK_WINDOW_HPP
#define GTK_WINDOW_HPP

// namespace Gtk {
// class Widget;
// }

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Window
 */
namespace Window {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC get_title( VMARG );

FALCON_FUNC set_title( VMARG );


} // Window
} // Gtk
} // Falcon

#endif // !GTK_WINDOW_HPP
