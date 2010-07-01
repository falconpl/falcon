#ifndef GTK_CELLRENDERERTEXT_HPP
#define GTK_CELLRENDERERTEXT_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERTEXT( item ) \
        ((GtkCellRendererText*)((Gtk::CellRendererText*) (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererText
 */
class CellRendererText
    :
    public Gtk::CoreGObject
{
public:

    CellRendererText( const Falcon::CoreClass*, const GtkCellRendererText* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_edited( VMARG );

    static void on_edited( GtkCellRendererText*, gchar*, gchar*, gpointer );

    static FALCON_FUNC set_fixed_height_from_font( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERTEXT_HPP
