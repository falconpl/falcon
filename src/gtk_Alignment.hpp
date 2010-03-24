#ifndef GTK_ALIGNMENT_HPP
#define GTK_ALIGNMENT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Alignment
 */
namespace Alignment {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC set( VMARG );

FALCON_FUNC get_padding( VMARG );

FALCON_FUNC set_padding( VMARG );


} // Alignment
} // Gtk
} // Falcon

#endif // !GTK_ALIGNMENT_HPP
