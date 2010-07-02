#ifndef GTK_CELLRENDERERSPINNER_HPP
#define GTK_CELLRENDERERSPINNER_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERSPINNER( item ) \
        ((GtkCellRendererSpinner*)((Gtk::CellRendererSpinner*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererSpinner
 */
class CellRendererSpinner
    :
    public Gtk::CoreGObject
{
public:

    CellRendererSpinner( const Falcon::CoreClass*, const GtkCellRendererSpinner* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERSPINNER_HPP
