#ifndef GTK_FIXED_HPP
#define GTK_FIXED_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Fixed
 */
namespace Fixed {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC put( VMARG );

FALCON_FUNC move( VMARG );

FALCON_FUNC get_has_window( VMARG );

FALCON_FUNC set_has_window( VMARG );


} // Fixed
} // Gtk
} // Falcon

#endif // !GTK_FIXED_HPP
