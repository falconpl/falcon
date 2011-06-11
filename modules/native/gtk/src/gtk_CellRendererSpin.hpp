#ifndef GTK_CELLRENDERERSPIN_HPP
#define GTK_CELLRENDERERSPIN_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERSPIN( item ) \
        ((GtkCellRendererSpin*)((Gtk::CellRendererSpin*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererSpin
 */
class CellRendererSpin
    :
    public Gtk::CoreGObject
{
public:

    CellRendererSpin( const Falcon::CoreClass*, const GtkCellRendererSpin* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERSPIN_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
