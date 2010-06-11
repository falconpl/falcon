#ifndef GTK_CELLRENDERER_HPP
#define GTK_CELLRENDERER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellRenderer
 */
class CellRenderer
    :
    public Gtk::CoreGObject
{
public:

    CellRenderer( const Falcon::CoreClass*, const GtkCellRenderer* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_editing_canceled( VMARG );

    static void on_editing_canceled( GtkCellRenderer*, gpointer );

    static FALCON_FUNC signal_editing_started( VMARG );

    static void on_editing_started( GtkCellRenderer*, GtkCellEditable*, gchar*, gpointer );

    static FALCON_FUNC get_size( VMARG );

    static FALCON_FUNC render( VMARG );

    static FALCON_FUNC activate( VMARG );

    static FALCON_FUNC start_editing( VMARG );

#if 0 // deprecated
    static FALCON_FUNC editing_canceled( VMARG );
#endif

    static FALCON_FUNC stop_editing( VMARG );

    static FALCON_FUNC get_fixed_size( VMARG );

    static FALCON_FUNC set_fixed_size( VMARG );

#if GTK_MINOR_VERSION >= 18
    static FALCON_FUNC get_visible( VMARG );

    static FALCON_FUNC set_visible( VMARG );

    static FALCON_FUNC get_sensitive( VMARG );

    static FALCON_FUNC set_sensitive( VMARG );

    static FALCON_FUNC get_alignment( VMARG );

    static FALCON_FUNC set_alignment( VMARG );

    static FALCON_FUNC get_padding( VMARG );

    static FALCON_FUNC set_padding( VMARG );
#endif // GTK_MINOR_VERSION >= 18

};


} // Gtk
} // Falcon

#endif // !GTK_CELLRENDERER_HPP
