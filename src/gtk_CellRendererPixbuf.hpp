#ifndef GTK_CELLRENDERERPIXBUF_HPP
#define GTK_CELLRENDERERPIXBUF_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERPIXBUF( item ) \
        ((GtkCellRendererPixbuf*) Falcon::dyncast<Gtk::CellRendererPixbuf*>( (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererPixbuf
 */
class CellRendererPixbuf
    :
    public Gtk::CoreGObject
{
public:

    CellRendererPixbuf( const Falcon::CoreClass*, const GtkCellRendererPixbuf* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERPIXBUF_HPP
