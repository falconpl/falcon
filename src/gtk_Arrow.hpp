#ifndef GTK_ARROW_HPP
#define GTK_ARROW_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Arrow
 */
namespace Arrow {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC set( VMARG );


} // Arrow
} // Gtk
} // Falcon

#endif // !GTK_ARROW_HPP
