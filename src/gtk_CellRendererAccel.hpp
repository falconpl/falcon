#ifndef GTK_CELLRENDERERACCEL_HPP
#define GTK_CELLRENDERERACCEL_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERACCEL( item ) \
        ((GtkCellRendererAccel*)((Gtk::CellRendererAccel*) (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererAccel
 */
class CellRendererAccel
    :
    public Gtk::CoreGObject
{
public:

    CellRendererAccel( const Falcon::CoreClass*, const GtkCellRendererAccel* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_accel_cleared( VMARG );

    static void on_accel_cleared( GtkCellRendererAccel*, gchar*, gpointer );

    static FALCON_FUNC signal_accel_edited( VMARG );

    static void on_accel_edited( GtkCellRendererAccel*, gchar*, guint,
                                 GdkModifierType, guint, gpointer );

};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERACCEL_HPP
