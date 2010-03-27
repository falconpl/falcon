#ifndef GTK_TEXTTAGTABLE_HPP
#define GTK_TEXTTAGTABLE_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TextTagTable
 */
class TextTagTable
    :
    public Gtk::CoreGObject
{
public:

    TextTagTable( const Falcon::CoreClass*, const GtkTextTagTable* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_tag_added( VMARG );

    static void on_tag_added( GtkTextTagTable*, GtkTextTag*, gpointer );

    static FALCON_FUNC signal_tag_changed( VMARG );

    //static void on_tag_changed( GtkTextTagTable*, GtkTextTag*, gboolean, gpointer );

    static FALCON_FUNC signal_tag_removed( VMARG );

    static void on_tag_removed( GtkTextTagTable*, GtkTextTag*, gpointer );

    static FALCON_FUNC add( VMARG );

    static FALCON_FUNC remove( VMARG );

    static FALCON_FUNC lookup( VMARG );

    //static FALCON_FUNC foreach( VMARG );

    static FALCON_FUNC get_size( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TEXTTAGTABLE_HPP
