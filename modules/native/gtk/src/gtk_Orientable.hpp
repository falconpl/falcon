#ifndef GTK_ORIENTABLE_HPP
#define GTK_ORIENTABLE_HPP

#include "modgtk.hpp"

#if GTK_CHECK_VERSION( 2, 16, 0 )

namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Orientable
 */
namespace Orientable {

void clsInit( Falcon::Module*, Falcon::Symbol* );

FALCON_FUNC get_orientation( VMARG );

FALCON_FUNC set_orientation( VMARG );


} // Orientable
} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 16, 0 )

#endif // !GTK_ORIENTABLE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
