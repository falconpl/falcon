#ifndef GTK_RADIOBUTTON_HPP
#define GTK_RADIOBUTTON_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::RadioButton
 */
namespace RadioButton {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC signal_group_changed( VMARG );

void on_group_changed( GtkRadioButton*, gpointer );

//FALCON_FUNC get_group( VMARG );

//FALCON_FUNC set_group( VMARG );


} // RadioButton
} // Gtk
} // Falcon

#endif // !GTK_RADIOBUTTON_HPP
