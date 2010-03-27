#ifndef GTK_SPINBUTTON_HPP
#define GTK_SPINBUTTON_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>

namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::SpinButton
 */
namespace SpinButton {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC signal_change_value( VMARG );

void on_change_value( GtkSpinButton*, GtkScrollType, gpointer );

FALCON_FUNC signal_input( VMARG );

gint on_input( GtkSpinButton*, gpointer, gpointer );

FALCON_FUNC signal_output( VMARG );

gboolean on_output( GtkSpinButton*, gpointer );

FALCON_FUNC signal_value_changed( VMARG );

void on_value_changed( GtkSpinButton*, gpointer );

FALCON_FUNC signal_wrapped( VMARG );

void on_wrapped( GtkSpinButton*, gpointer );

//FALCON_FUNC set_adjustment( VMARG );

//FALCON_FUNC get_adjustment( VMARG );

FALCON_FUNC set_digits( VMARG );

FALCON_FUNC set_increments( VMARG );

FALCON_FUNC set_range( VMARG );

FALCON_FUNC get_value_as_int( VMARG );

FALCON_FUNC set_value( VMARG );

FALCON_FUNC set_update_policy( VMARG );

FALCON_FUNC set_numeric( VMARG );

FALCON_FUNC spin( VMARG );

FALCON_FUNC set_wrap( VMARG );

FALCON_FUNC set_snap_to_ticks( VMARG );

FALCON_FUNC update( VMARG );

FALCON_FUNC get_digits( VMARG );

FALCON_FUNC get_increments( VMARG );

FALCON_FUNC get_numeric( VMARG );

FALCON_FUNC get_range( VMARG );

FALCON_FUNC get_snap_to_ticks( VMARG );

FALCON_FUNC get_update_policy( VMARG );

FALCON_FUNC get_value( VMARG );

FALCON_FUNC get_wrap( VMARG );

} //SpinButton
} //Gtk
} //Falcon

#endif //!GTK_SPINBUTTON_HPP

