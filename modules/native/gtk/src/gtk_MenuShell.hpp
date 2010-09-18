#ifndef GTK_MENUSHELL_HPP
#define GTK_MENUSHELL_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::MenuShell
 */
class MenuShell
    :
    public Gtk::CoreGObject
{
public:

    MenuShell( const Falcon::CoreClass*, const GtkMenuShell* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC signal_activate_current( VMARG );

    static void on_activate_current( GtkMenuShell*, gboolean, gpointer );

    static FALCON_FUNC signal_cancel( VMARG );

    static void on_cancel( GtkMenuShell*, gpointer );

    static FALCON_FUNC signal_cycle_focus( VMARG );

    static void on_cycle_focus( GtkMenuShell*, GtkDirectionType, gpointer );

    static FALCON_FUNC signal_deactivate( VMARG );

    static void on_deactivate( GtkMenuShell*, gpointer );

    static FALCON_FUNC signal_move_current( VMARG );

    static void on_move_current( GtkMenuShell*, GtkMenuDirectionType, gpointer );

    static FALCON_FUNC signal_move_selected( VMARG );

    static gboolean on_move_selected( GtkMenuShell*, gint, gpointer );

    static FALCON_FUNC signal_selection_done( VMARG );

    static void on_selection_done( GtkMenuShell*, gpointer );

    static FALCON_FUNC append( VMARG );

    static FALCON_FUNC prepend( VMARG );

    static FALCON_FUNC insert( VMARG );

    static FALCON_FUNC deactivate( VMARG );

    static FALCON_FUNC select_item( VMARG );

    static FALCON_FUNC select_first( VMARG );

    static FALCON_FUNC deselect( VMARG );

    static FALCON_FUNC activate_item( VMARG );

    static FALCON_FUNC cancel( VMARG );

    static FALCON_FUNC set_take_focus( VMARG );

    static FALCON_FUNC get_take_focus( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_MENUSHELL_HPP
