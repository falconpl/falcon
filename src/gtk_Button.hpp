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

FALCON_FUNC signal_activate( VMARG );

void on_activate( GtkButton*, gpointer );

FALCON_FUNC signal_clicked( VMARG );

void on_clicked( GtkButton*, gpointer );

FALCON_FUNC signal_enter( VMARG );

void on_enter( GtkButton*, gpointer );

FALCON_FUNC signal_leave( VMARG );

void on_leave( GtkButton*, gpointer );

FALCON_FUNC signal_pressed( VMARG );

void on_pressed( GtkButton*, gpointer );

FALCON_FUNC signal_released( VMARG );

void on_released( GtkButton*, gpointer );

FALCON_FUNC pressed( VMARG );

FALCON_FUNC released( VMARG );

FALCON_FUNC clicked( VMARG );

FALCON_FUNC enter( VMARG );

FALCON_FUNC leave( VMARG );

//FALCON_FUNC set_relief( VMARG );

//FALCON_FUNC get_relief( VMARG );

FALCON_FUNC set_label( VMARG );

FALCON_FUNC get_label( VMARG );

FALCON_FUNC set_use_stock( VMARG );

FALCON_FUNC get_use_stock( VMARG );

FALCON_FUNC set_use_underline( VMARG );

FALCON_FUNC get_use_underline( VMARG );

FALCON_FUNC set_focus_on_click( VMARG );

FALCON_FUNC get_focus_on_click( VMARG );

//FALCON_FUNC set_alignment( VMARG );

//FALCON_FUNC get_alignment( VMARG );

FALCON_FUNC set_image( VMARG );

FALCON_FUNC get_image( VMARG );

//FALCON_FUNC set_image_position( VMARG );

//FALCON_FUNC get_image_position( VMARG );


} // Button
} // Gtk
} // Falcon

#endif // !GTK_BUTTON_HPP
