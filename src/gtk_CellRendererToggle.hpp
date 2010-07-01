#ifndef GTK_CELLRENDERERTOGGLE_HPP
#define GTK_CELLRENDERERTOGGLE_HPP

#include "modgtk.hpp"

#define GET_CELLRENDERERTOGGLE( item ) \
        ((GtkCellRendererToggle*)((Gtk::CellRendererToggle*) (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRendererToggle
 */
class CellRendererToggle
    :
    public Gtk::CoreGObject
{
public:

    CellRendererToggle( const Falcon::CoreClass*, const GtkCellRendererToggle* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_toggled( VMARG );

    static void on_toggled( GtkCellRendererToggle*, gchar*, gpointer );

    static FALCON_FUNC get_radio( VMARG );

    static FALCON_FUNC set_radio( VMARG );

    static FALCON_FUNC get_active( VMARG );

    static FALCON_FUNC set_active( VMARG );

#if GTK_CHECK_VERSION( 2, 18, 0 )
    static FALCON_FUNC get_activatable( VMARG );

    static FALCON_FUNC set_activatable( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERERTOGGLE_HPP
