#ifndef GTK_INFOBAR_HPP
#define GTK_INFOBAR_HPP

#include "modgtk.hpp"

#if GTK_CHECK_VERSION( 2, 18, 0 )

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::InfoBar
 */
class InfoBar
    :
    public Gtk::CoreGObject
{
public:

    InfoBar( const Falcon::CoreClass*, const GtkInfoBar* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC new_with_buttons( VMARG );

    static FALCON_FUNC add_action_widget( VMARG );

    static FALCON_FUNC add_button( VMARG );

    //static FALCON_FUNC add_buttons( VMARG );

    static FALCON_FUNC set_response_sensitive( VMARG );

    static FALCON_FUNC set_default_response( VMARG );

    static FALCON_FUNC response( VMARG );

    static FALCON_FUNC set_message_type( VMARG );

    static FALCON_FUNC get_message_type( VMARG );

    static FALCON_FUNC get_action_area( VMARG );

    static FALCON_FUNC get_content_area( VMARG );

};


} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 18, 0 )
#endif // !GTK_INFOBAR_HPP
