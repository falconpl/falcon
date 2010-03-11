#ifndef GTK_WIDGET_HPP
#define GTK_WIDGET_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Widget
 */
class Widget
    :
    public Falcon::CoreObject
{
public:

    Widget( const Falcon::CoreClass*, const GtkWidget* = 0 );

    ~Widget() {}

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_delete_event( VMARG );

    static gboolean on_delete_event( GtkWidget*, GdkEvent*, gpointer );

    static FALCON_FUNC signal_show( VMARG );

    static void on_show( GtkWidget*, GdkEvent*, gpointer );

    static FALCON_FUNC signal_hide( VMARG );

    static void on_hide( GtkWidget*, GdkEvent*, gpointer );

    static FALCON_FUNC show( VMARG );

    static FALCON_FUNC show_now( VMARG );

    static FALCON_FUNC hide( VMARG );

    static FALCON_FUNC show_all( VMARG );

    static FALCON_FUNC hide_all( VMARG );

    static FALCON_FUNC activate( VMARG );

    static FALCON_FUNC reparent( VMARG );

    static FALCON_FUNC is_focus( VMARG );

    static FALCON_FUNC grab_focus( VMARG );

    static FALCON_FUNC grab_default( VMARG );

    static FALCON_FUNC set_name( VMARG );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC set_sensitive( VMARG );

    static FALCON_FUNC get_toplevel( VMARG );

    static FALCON_FUNC get_events( VMARG );

    static FALCON_FUNC is_ancestor( VMARG );

    static FALCON_FUNC hide_on_delete( VMARG );

};

} // Gtk
} // Falcon

#endif // !GTK_WIDGET_HPP
