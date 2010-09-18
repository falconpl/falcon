#ifndef GTK_CELLRENDERERPROGRESS_HPP
#define GTK_CELLRENDERERPROGRESS_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERPROGRESS( item ) \
        ((GtkCellRendererProgress*)((Gtk::CellRendererProgress*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererProgress
 */
class CellRendererProgress
    :
    public Gtk::CoreGObject
{
public:

    CellRendererProgress( const Falcon::CoreClass*, const GtkCellRendererProgress* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERPROGRESS_HPP
