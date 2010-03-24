#ifndef GTK_ASPECTFRAME_HPP
#define GTK_ASPECTFRAME_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::AspectFrame
 */
namespace AspectFrame {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC set( VMARG );


} // AspectFrame
} // Gtk
} // Falcon

#endif // !GTK_ASPECTFRAME_HPP
