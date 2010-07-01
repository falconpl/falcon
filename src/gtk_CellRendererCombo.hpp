#ifndef GTK_CELLRENDERERCOMBO_HPP
#define GTK_CELLRENDERERCOMBO_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERCOMBO( item ) \
        ((GtkCellRendererCombo*)((Gtk::CellRendererCombo*) (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererCombo
 */
class CellRendererCombo
    :
    public Gtk::CoreGObject
{
public:

    CellRendererCombo( const Falcon::CoreClass*, const GtkCellRendererCombo* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC signal_changed( VMARG );

    static void on_changed( GtkCellRendererCombo*, gchar*, GtkTreeIter*, gpointer );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERCOMBO_HPP
