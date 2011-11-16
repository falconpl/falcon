/**
 *  \file gtk_Accelerator.hpp
 */

#ifndef GTK_ACCELERATOR_HPP
#define GTK_ACCELERATOR_HPP

#include "modgtk.hpp"

namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Accelerator
 */
namespace Accelerator {

void modInit( Falcon::Module* );

FALCON_FUNC parse( VMARG );

} // Accelerator
} // Gtk
} // Falcon

#endif // !GTK_ACCELERATOR_HPP

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
