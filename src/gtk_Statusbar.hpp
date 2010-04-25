#ifndef GTK_STATUSBAR_HPP
#define GTK_STATUSBAR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Statusbar
 */
class Statusbar
    :
    public Gtk::CoreGObject
{
public:

    Statusbar( const Falcon::CoreClass*, const GtkStatusbar* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_text_popped( VMARG );

    static void on_text_popped( GtkStatusbar*, guint, gchar*, gpointer );

    static FALCON_FUNC signal_text_pushed( VMARG );

    static void on_text_pushed( GtkStatusbar*, guint, gchar*, gpointer );

    static FALCON_FUNC get_context_id( VMARG );

    static FALCON_FUNC push( VMARG );

    static FALCON_FUNC pop( VMARG );

    static FALCON_FUNC remove( VMARG );

    static FALCON_FUNC set_has_resize_grip( VMARG );

    static FALCON_FUNC get_has_resize_grip( VMARG );

#if GTK_MINOR_VERSION >= 20
    static FALCON_FUNC get_message_area( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_STATUSBAR_HPP
