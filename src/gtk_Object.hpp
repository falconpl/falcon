#ifndef GTK_OBJECT_HPP
#define GTK_OBJECT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Object
 *  \note Most of its C functions/macros are deprecated in favor of equivalent
 *  GObject functions/macros.
 */
namespace Object {

void modInit( Falcon::Module* );

FALCON_FUNC signal_destroy( VMARG );

void on_destroy( GObject*, gpointer );

FALCON_FUNC destroy( VMARG );


} // Object
} // Gtk
} // Falcon

#endif // !GTK_OBJECT_HPP
