#ifndef GTK_ADJUSTMENT_HPP
#define GTK_ADJUSTMENT_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Adjustment
 */
namespace Adjustment {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC signal_changed( VMARG );

void on_changed( GtkAdjustment*, gpointer );

FALCON_FUNC signal_value_changed( VMARG );

void on_value_changed( GtkAdjustment*, gpointer );

FALCON_FUNC get_value( VMARG );

FALCON_FUNC set_value( VMARG );

FALCON_FUNC clamp_page( VMARG );

FALCON_FUNC changed( VMARG );

FALCON_FUNC value_changed( VMARG );

FALCON_FUNC configure( VMARG );

FALCON_FUNC get_lower( VMARG );

FALCON_FUNC get_page_increment( VMARG );

FALCON_FUNC get_page_size( VMARG );

FALCON_FUNC get_step_increment( VMARG );

FALCON_FUNC get_upper( VMARG );

FALCON_FUNC set_lower( VMARG );

FALCON_FUNC set_page_increment( VMARG );

FALCON_FUNC set_page_size( VMARG );

FALCON_FUNC set_step_increment( VMARG );

FALCON_FUNC set_upper( VMARG );


} // Adjustment
} // Gtk
} // Falcon

#endif // !GTK_ADJUSTMENT_HPP
