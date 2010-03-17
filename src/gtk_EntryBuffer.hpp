#ifndef GTK_ENTRYBUFFER_HPP
#define GTK_ENTRYBUFFER_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>

#if GTK_MINOR_VERSION >= 18

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::EntryBuffer
 */
class EntryBuffer
    :
    public Gtk::CoreGObject
{
public:

    EntryBuffer( const Falcon::CoreClass*, const GtkEntryBuffer* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 18

#endif // !GTK_ENTRYBUFFER_HPP
