#ifndef GTK_DIALOG_HPP
#define GTK_DIALOG_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Dialog
 */
class Dialog
    :
    public Gtk::CoreGObject
{
public:

    Dialog( const Falcon::CoreClass*, const GtkDialog* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC new_with_buttons( VMARG );

    static FALCON_FUNC run( VMARG );

    static FALCON_FUNC response( VMARG );

    static FALCON_FUNC add_button( VMARG );

    //static FALCON_FUNC add_buttons( VMARG );

    static FALCON_FUNC add_action_widget( VMARG );

    static FALCON_FUNC get_has_separator( VMARG );

    static FALCON_FUNC set_default_response( VMARG );

    static FALCON_FUNC set_has_separator( VMARG );

    static FALCON_FUNC set_response_sensitive( VMARG );

#if GTK_MINOR_VERSION >= 8
    static FALCON_FUNC get_response_for_widget( VMARG );
#endif

#if GTK_MINOR_VERSION >= 20
    static FALCON_FUNC get_widget_for_response( VMARG );
#endif

#if GTK_MINOR_VERSION >= 14
    static FALCON_FUNC get_action_area( VMARG );

    static FALCON_FUNC get_content_area( VMARG );
#endif

#if GTK_MINOR_VERSION >= 6
    //static FALCON_FUNC alternative_dialog_button_order( VMARG );

    //static FALCON_FUNC set_alternative_button_order( VMARG );

    //static FALCON_FUNC set_alternative_button_order_from_array( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_DIALOG_HPP
