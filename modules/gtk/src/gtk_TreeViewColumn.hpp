#ifndef GTK_TREEVIEWCOLUMN_HPP
#define GTK_TREEVIEWCOLUMN_HPP

#include "modgtk.hpp"

#define GET_TREEVIEWCOLUMN( item ) \
        ((GtkTreeViewColumn*)((Gtk::TreeViewColumn*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeViewColumn
 */
class TreeViewColumn
    :
    public Gtk::CoreGObject
{
public:

    TreeViewColumn( const Falcon::CoreClass*, const GtkTreeViewColumn* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_clicked( VMARG );

    static void on_clicked( GtkTreeViewColumn*, gpointer );

    static FALCON_FUNC new_with_attributes( VMARG );

    static FALCON_FUNC pack_start( VMARG );

    static FALCON_FUNC pack_end( VMARG );

    static FALCON_FUNC clear( VMARG );

    static FALCON_FUNC add_attribute( VMARG );

    static FALCON_FUNC set_attributes( VMARG );

    static FALCON_FUNC set_cell_data_func( VMARG );

    static void exec_cell_data_func( GtkTreeViewColumn*, GtkCellRenderer*,
                                     GtkTreeModel*, GtkTreeIter*, gpointer );

    static FALCON_FUNC clear_attributes( VMARG );

    static FALCON_FUNC set_spacing( VMARG );

    static FALCON_FUNC get_spacing( VMARG );

    static FALCON_FUNC set_visible( VMARG );

    static FALCON_FUNC get_visible( VMARG );

    static FALCON_FUNC set_resizable( VMARG );

    static FALCON_FUNC get_resizable( VMARG );

    static FALCON_FUNC set_sizing( VMARG );

    static FALCON_FUNC get_sizing( VMARG );

    static FALCON_FUNC get_width( VMARG );

    static FALCON_FUNC get_fixed_width( VMARG );

    static FALCON_FUNC set_fixed_width( VMARG );

    static FALCON_FUNC set_min_width( VMARG );

    static FALCON_FUNC get_min_width( VMARG );

    static FALCON_FUNC set_max_width( VMARG );

    static FALCON_FUNC get_max_width( VMARG );

    static FALCON_FUNC clicked( VMARG );

    static FALCON_FUNC set_title( VMARG );

    static FALCON_FUNC get_title( VMARG );

    static FALCON_FUNC set_expand( VMARG );

    static FALCON_FUNC get_expand( VMARG );

    static FALCON_FUNC set_clickable( VMARG );

    static FALCON_FUNC get_clickable( VMARG );

    static FALCON_FUNC set_widget( VMARG );

    static FALCON_FUNC get_widget( VMARG );

    static FALCON_FUNC set_alignment( VMARG );

    static FALCON_FUNC get_alignment( VMARG );

    static FALCON_FUNC set_reorderable( VMARG );

    static FALCON_FUNC get_reorderable( VMARG );

    static FALCON_FUNC set_sort_column_id( VMARG );

    static FALCON_FUNC get_sort_column_id( VMARG );

    static FALCON_FUNC set_sort_indicator( VMARG );

    static FALCON_FUNC get_sort_indicator( VMARG );

    static FALCON_FUNC set_sort_order( VMARG );

    static FALCON_FUNC get_sort_order( VMARG );

    static FALCON_FUNC cell_set_cell_data( VMARG );

    static FALCON_FUNC cell_get_size( VMARG );

    static FALCON_FUNC cell_get_position( VMARG );

    static FALCON_FUNC cell_is_visible( VMARG );

    static FALCON_FUNC focus_cell( VMARG );

    static FALCON_FUNC queue_resize( VMARG );

    static FALCON_FUNC get_tree_view( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TREEVIEWCOLUMN_HPP
