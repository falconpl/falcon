#ifndef GTK_CELLEDITABLE_HPP
#define GTK_CELLEDITABLE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CellEditable
 *  \note This is both an interface and a class.
 */
class CellEditable
    :
    public Gtk::CoreGObject
{
public:

    CellEditable( const Falcon::CoreClass*, const GtkCellEditable* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static void clsInit( Falcon::Module*, Falcon::Symbol* );

    static FALCON_FUNC signal_editing_done( VMARG );

    static void on_editing_done( GtkCellEditable*, gpointer );

    static FALCON_FUNC signal_remove_widget( VMARG );

    static void on_remove_widget( GtkCellEditable*, gpointer );

    static FALCON_FUNC start_editing( VMARG );

    static FALCON_FUNC editing_done( VMARG );

    static FALCON_FUNC remove_widget( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CELLEDITABLE_HPP
