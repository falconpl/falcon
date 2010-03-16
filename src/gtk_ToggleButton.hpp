#ifndef GTK_TOGGLEBUTTON_HPP
#define GTK_TOGGLEBUTTON_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::ToggleButton
 */
namespace ToggleButton {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC signal_toggled( VMARG );

void on_toggled( GtkToggleButton*, gpointer );

FALCON_FUNC set_mode( VMARG );

FALCON_FUNC get_mode( VMARG );

FALCON_FUNC toggled( VMARG );

FALCON_FUNC get_active( VMARG );

FALCON_FUNC set_active( VMARG );

FALCON_FUNC get_inconsistent( VMARG );

FALCON_FUNC set_inconsistent( VMARG );


} // ToggleButton
} // Gtk
} // Falcon

#endif // !GTK_TOGGLEBUTTON_HPP
