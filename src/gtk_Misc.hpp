#ifndef GTK_MISC_HPP
#define GTK_MISC_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Misc
 */
namespace Misc {

void modInit( Falcon::Module* );

FALCON_FUNC set_alignment( VMARG );

FALCON_FUNC set_padding( VMARG );

FALCON_FUNC get_alignment( VMARG );

FALCON_FUNC get_padding( VMARG );


} // Misc
} // Gtk
} // Falcon

#endif // !GTK_MISC_HPP
