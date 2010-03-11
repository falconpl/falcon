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

void get_title( PROPARG );

void set_title( PROPARG );


} // Window
} // Gtk
} // Falcon

#endif // !GTK_WINDOW_HPP
