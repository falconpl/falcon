#ifndef GTK_OBJECT_HPP
#define GTK_OBJECT_HPP

#include "modgtk.hpp"

#define GET_GTKOBJECT( item ) \
        ((GtkObject*)((Gtk::Object*) (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Object
 *  \note Most of its C functions/macros are deprecated in favor of equivalent
 *  GObject functions/macros.
 */
class Object
    :
    public Gtk::CoreGObject
{
public:

    Object( const Falcon::CoreClass*, const GtkObject* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_destroy( VMARG );

    static void on_destroy( GObject*, gpointer );

    static FALCON_FUNC destroy( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_OBJECT_HPP
