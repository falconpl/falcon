#ifndef GTK_BIN_HPP
#define GTK_BIN_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Bin
 */
namespace Bin {

void modInit( Falcon::Module* );

FALCON_FUNC get_child( VMARG );


} // Bin
} // Gtk
} // Falcon

#endif // !GTK_BIN_HPP
