#ifndef GTK_TREEMODEL_HPP
#define GTK_TREEMODEL_HPP

#include "modgtk.hpp"

#define GET_TREEMODEL( item ) \
        ((GtkTreeModel*)((Gtk::TreeModel*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeModel
 *  \note This is both an interface and a class.
 */
class TreeModel
    :
    public Gtk::CoreGObject
{
public:

    TreeModel( const Falcon::CoreClass*, const GtkTreeModel* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static void clsInit( Falcon::Module*, Falcon::Symbol* );

    static bool implementedBy( const Falcon::Item* );

    static FALCON_FUNC signal_row_changed( VMARG );

    static void on_row_changed( GtkTreeModel*, GtkTreePath*, GtkTreeIter*, gpointer );

    static FALCON_FUNC signal_row_deleted( VMARG );

    static void on_row_deleted( GtkTreeModel*, GtkTreePath*, gpointer );

    static FALCON_FUNC signal_row_has_child_toggled( VMARG );

    static void on_row_has_child_toggled( GtkTreeModel*, GtkTreePath*, GtkTreeIter*, gpointer );

    static FALCON_FUNC signal_row_inserted( VMARG );

    static void on_row_inserted( GtkTreeModel*, GtkTreePath*, GtkTreeIter*, gpointer );

    static FALCON_FUNC signal_rows_reordered( VMARG );

    static void on_rows_reordered( GtkTreeModel*, GtkTreePath*, GtkTreeIter*, gpointer, gpointer );

    static FALCON_FUNC get_flags( VMARG );

    static FALCON_FUNC get_n_columns( VMARG );

    static FALCON_FUNC get_column_type( VMARG );

    static FALCON_FUNC get_iter( VMARG );

    static FALCON_FUNC get_iter_from_string( VMARG );

    static FALCON_FUNC get_iter_first( VMARG );

    static FALCON_FUNC get_path( VMARG );

    static FALCON_FUNC get_value( VMARG );

    static FALCON_FUNC iter_next( VMARG );

    static FALCON_FUNC iter_children( VMARG );

    static FALCON_FUNC iter_has_child( VMARG );

    static FALCON_FUNC iter_n_children( VMARG );

    static FALCON_FUNC iter_nth_child( VMARG );

    static FALCON_FUNC iter_parent( VMARG );

    static FALCON_FUNC get_string_from_iter( VMARG );

#if 0 // todo
    static FALCON_FUNC ref_node( VMARG );
    static FALCON_FUNC unref_node( VMARG );
    static FALCON_FUNC get( VMARG );
    static FALCON_FUNC get_valist( VMARG );
    static FALCON_FUNC foreach_( VMARG );
#endif

    static FALCON_FUNC row_changed( VMARG );

    static FALCON_FUNC row_inserted( VMARG );

    static FALCON_FUNC row_has_child_toggled( VMARG );

    static FALCON_FUNC row_deleted( VMARG );

    static FALCON_FUNC rows_reordered( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TREEMODEL_HPP
