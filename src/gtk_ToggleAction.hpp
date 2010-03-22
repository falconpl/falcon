#ifndef GTK_TOGGLEACTION_HPP
#define GTK_TOGGLEACTION_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::ToggleAction
 */
namespace ToggleAction {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC signal_toggled( VMARG );

void on_toggled( GtkToggleAction*, gpointer );

FALCON_FUNC toggled( VMARG );

FALCON_FUNC set_active( VMARG );

FALCON_FUNC get_active( VMARG );

FALCON_FUNC set_draw_as_radio( VMARG );

FALCON_FUNC get_draw_as_radio( VMARG );


} // ToggleAction
} // Gtk
} // Falcon

#endif // !GTK_TOGGLEACTION_HPP
