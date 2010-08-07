#ifndef GTK_MAIN_HPP
#define GTK_MAIN_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Main
 */
namespace Main {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC quit( VMARG );

FALCON_FUNC run( VMARG );


} // Main
} // Gtk
} // Falcon

#endif // !GTK_MAIN_HPP
