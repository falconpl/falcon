#ifndef GTK_TEXTBUFFER_HPP
#define GTK_TEXTBUFFER_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TextBuffer
 */
class TextBuffer
    :
    public Gtk::CoreGObject
{
public:

    TextBuffer( const Falcon::CoreClass*, const GtkTextBuffer* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );


};


} // Gtk
} // Falcon

#endif // !GTK_TEXTBUFFER_HPP
