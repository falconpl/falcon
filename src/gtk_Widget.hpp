#ifndef GTK_WIDGET_HPP
#define GTK_WIDGET_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Widget
 */
namespace Widget {

void modInit( Falcon::Module* );

FALCON_FUNC signal_delete_event( VMARG );

gboolean on_delete_event( GtkWidget*, GdkEvent*, gpointer );

FALCON_FUNC signal_show( VMARG );

void on_show( GtkWidget*, GdkEvent*, gpointer );

FALCON_FUNC signal_hide( VMARG );

void on_hide( GtkWidget*, GdkEvent*, gpointer );

FALCON_FUNC show( VMARG );

FALCON_FUNC show_now( VMARG );

FALCON_FUNC hide( VMARG );

FALCON_FUNC show_all( VMARG );

FALCON_FUNC hide_all( VMARG );

FALCON_FUNC activate( VMARG );


} // Widget
} // Gtk
} // Falcon

#endif // !GTK_WIDGET_HPP
