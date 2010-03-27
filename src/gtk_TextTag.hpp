#ifndef GTK_TEXTTAG_HPP
#define GTK_TEXTTAG_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TextTag
 */
class TextTag
    :
    public Gtk::CoreGObject
{
public:

    TextTag( const Falcon::CoreClass*, const GtkTextTag* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC signal_event( VMARG );

    static FALCON_FUNC get_priority( VMARG );

    static FALCON_FUNC set_priority( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TEXTTAG_HPP
