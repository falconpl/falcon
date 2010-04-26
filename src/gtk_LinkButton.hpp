#ifndef GTK_LINKBUTTON_HPP
#define GTK_LINKBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::LinkButton
 */
class LinkButton
    :
    public Gtk::CoreGObject
{
public:

    LinkButton( const Falcon::CoreClass*, const GtkLinkButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC get_uri( VMARG );

    static FALCON_FUNC set_uri( VMARG );

    //static FALCON_FUNC set_uri_hook( VMARG );

#if GTK_MINOR_VERSION >= 14
    static FALCON_FUNC get_visited( VMARG );

    static FALCON_FUNC set_visited( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_LINKBUTTON_HPP
