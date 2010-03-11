#ifndef GTK_BUTTON_HPP
#define GTK_BUTTON_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Button
 */
namespace Button {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC signal_clicked( VMARG );

void on_clicked( GtkButton*, gpointer );


} // Button
} // Gtk
} // Falcon

#endif // !GTK_BUTTON_HPP
