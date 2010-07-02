#ifndef GTK_TREESORTABLE_HPP
#define GTK_TREESORTABLE_HPP

#include "modgtk.hpp"

#define GET_TREESORTABLE( item ) \
        ((GtkTreeSortable*)((Gtk::TreeSortable*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeSortable
 *  \note This is both an interface and a class.
 */
class TreeSortable
    :
    public Gtk::CoreGObject
{
public:

    TreeSortable( const Falcon::CoreClass*, const GtkTreeSortable* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static void clsInit( Falcon::Module*, Falcon::Symbol* );

    static FALCON_FUNC signal_sort_column_changed( VMARG );

    static void on_sort_column_changed( GtkTreeSortable*, gpointer );

    static FALCON_FUNC sort_column_changed( VMARG );

    static FALCON_FUNC get_sort_column_id( VMARG );

    static FALCON_FUNC set_sort_column_id( VMARG );

    #if 0 // todo
    static FALCON_FUNC set_sort_func( VMARG );
    static FALCON_FUNC set_default_sort_func( VMARG );
#endif

    static FALCON_FUNC has_default_sort_func( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TREESORTABLE_HPP
