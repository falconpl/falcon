#ifndef GTK_CELLRENDERERSPINNER_HPP
#define GTK_CELLRENDERERSPINNER_HPP

#include "modgtk.hpp"

#if GTK_CHECK_VERSION( 2, 20, 0 )

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

#endif // GTK_CHECK_VERSION( 2, 20, 0 )

#endif // !GTK_CELLRENDERERSPINNER_HPP
