#ifndef GTK_TREEVIEW_HPP
#define GTK_TREEVIEW_HPP

#include "modgtk.hpp"

#define GET_TREEVIEW( item ) \
        ((GtkTreeView*)((Gtk::TreeView*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeView
 */
class TreeView
    :
    public Gtk::CoreGObject
{
public:

    TreeView( const Falcon::CoreClass*, const GtkTreeView* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_columns_changed( VMARG );

    static void on_columns_changed( GtkTreeView*, gpointer );

    static FALCON_FUNC signal_cursor_changed( VMARG );

    static void on_cursor_changed( GtkTreeView*, gpointer );

    static FALCON_FUNC signal_expand_collapse_cursor_row( VMARG );

    static gboolean on_expand_collapse_cursor_row( GtkTreeView*, gboolean, gboolean, gboolean, gpointer );

    static FALCON_FUNC signal_move_cursor( VMARG );

    static gboolean on_move_cursor( GtkTreeView*, GtkMovementStep, gint, gpointer );

    static FALCON_FUNC signal_row_activated( VMARG );

    static void on_row_activated( GtkTreeView*, GtkTreePath*, GtkTreeViewColumn*, gpointer );

    static FALCON_FUNC signal_row_collapsed( VMARG );

    static void on_row_collapsed( GtkTreeView*, GtkTreeIter*, GtkTreePath*, gpointer );

    static FALCON_FUNC signal_row_expanded( VMARG );

    static void on_row_expanded( GtkTreeView*, GtkTreeIter*, GtkTreePath*, gpointer );

    static FALCON_FUNC signal_select_all( VMARG );

    static gboolean on_select_all( GtkTreeView*, gpointer );

    static FALCON_FUNC signal_select_cursor_parent( VMARG );

    static gboolean on_select_cursor_parent( GtkTreeView*, gpointer );

    static FALCON_FUNC signal_select_cursor_row( VMARG );

    static gboolean on_select_cursor_row( GtkTreeView*, gboolean, gpointer );

    static FALCON_FUNC signal_set_scroll_adjustments( VMARG );

    static void on_set_scroll_adjustments( GtkTreeView*, GtkAdjustment*, GtkAdjustment*, gpointer );

    static FALCON_FUNC signal_start_interactive_search( VMARG );

    static gboolean on_start_interactive_search( GtkTreeView*, gpointer );

    static FALCON_FUNC signal_test_collapse_row( VMARG );

    static gboolean on_test_collapse_row( GtkTreeView*, GtkTreeIter*, GtkTreePath*, gpointer );

    static FALCON_FUNC signal_test_expand_row( VMARG );

    static gboolean on_test_expand_row( GtkTreeView*, GtkTreeIter*, GtkTreePath*, gpointer );

    static FALCON_FUNC signal_toggle_cursor_row( VMARG );

    static gboolean on_toggle_cursor_row( GtkTreeView*, gpointer );

    static FALCON_FUNC signal_unselect_all( VMARG );

    static gboolean on_unselect_all( GtkTreeView*, gpointer );

    static FALCON_FUNC get_level_indentation( VMARG );

    static FALCON_FUNC get_show_expanders( VMARG );

    static FALCON_FUNC set_level_indentation( VMARG );

    static FALCON_FUNC set_show_expanders( VMARG );

    static FALCON_FUNC new_with_model( VMARG );

    static FALCON_FUNC get_model( VMARG );

    static FALCON_FUNC set_model( VMARG );
#if 0 // todo
    static FALCON_FUNC get_selection( VMARG );
#endif
    static FALCON_FUNC get_hadjustment( VMARG );

    static FALCON_FUNC set_hadjustment( VMARG );

    static FALCON_FUNC get_vadjustment( VMARG );

    static FALCON_FUNC set_vadjustment( VMARG );

    static FALCON_FUNC get_headers_visible( VMARG );

    static FALCON_FUNC set_headers_visible( VMARG );

    static FALCON_FUNC columns_autosize( VMARG );

    static FALCON_FUNC get_headers_clickable( VMARG );

    static FALCON_FUNC set_headers_clickable( VMARG );

    static FALCON_FUNC set_rules_hint( VMARG );

    static FALCON_FUNC get_rules_hint( VMARG );

    static FALCON_FUNC append_column( VMARG );

    static FALCON_FUNC remove_column( VMARG );

    static FALCON_FUNC insert_column( VMARG );

    static FALCON_FUNC insert_column_with_attributes( VMARG );

    static FALCON_FUNC insert_column_with_data_func( VMARG );

    static FALCON_FUNC get_column( VMARG );

    static FALCON_FUNC get_columns( VMARG );

    static FALCON_FUNC move_column_after( VMARG );

    static FALCON_FUNC set_expander_column( VMARG );

    static FALCON_FUNC get_expander_column( VMARG );

    static FALCON_FUNC set_column_drag_function( VMARG );

    static gboolean exec_column_drag_function( GtkTreeView*, GtkTreeViewColumn*, GtkTreeViewColumn*, GtkTreeViewColumn*, gpointer );

    static FALCON_FUNC scroll_to_point( VMARG );

    static FALCON_FUNC scroll_to_cell( VMARG );

    static FALCON_FUNC set_cursor( VMARG );

    static FALCON_FUNC set_cursor_on_cell( VMARG );

    static FALCON_FUNC get_cursor( VMARG );

    static FALCON_FUNC row_activated( VMARG );

    static FALCON_FUNC expand_all( VMARG );

    static FALCON_FUNC collapse_all( VMARG );

    static FALCON_FUNC expand_to_path( VMARG );

    static FALCON_FUNC expand_row( VMARG );

    static FALCON_FUNC collapse_row( VMARG );
#if 0
    static FALCON_FUNC map_expanded_rows( VMARG );
    static FALCON_FUNC row_expanded( VMARG );
    static FALCON_FUNC set_reorderable( VMARG );
    static FALCON_FUNC get_reorderable( VMARG );
    static FALCON_FUNC get_path_at_pos( VMARG );
    static FALCON_FUNC get_cell_area( VMARG );
    static FALCON_FUNC get_background_area( VMARG );
    static FALCON_FUNC get_visible_rect( VMARG );
    static FALCON_FUNC get_visible_range( VMARG );
    static FALCON_FUNC get_bin_window( VMARG );
    static FALCON_FUNC convert_bin_window_to_tree_coords( VMARG );
    static FALCON_FUNC convert_bin_window_to_widget_coords( VMARG );
    static FALCON_FUNC convert_tree_to_bin_window_coords( VMARG );
    static FALCON_FUNC convert_tree_to_widget_coords( VMARG );
    static FALCON_FUNC convert_widget_to_bin_window_coords( VMARG );
    static FALCON_FUNC convert_widget_to_tree_coords( VMARG );
    static FALCON_FUNC enable_model_drag_dest( VMARG );
    static FALCON_FUNC enable_model_drag_source( VMARG );
    static FALCON_FUNC unset_rows_drag_source( VMARG );
    static FALCON_FUNC unset_rows_drag_dest( VMARG );
    static FALCON_FUNC set_drag_dest_row( VMARG );
    static FALCON_FUNC get_drag_dest_row( VMARG );
    static FALCON_FUNC get_dest_row_at_pos( VMARG );
    static FALCON_FUNC create_row_drag_icon( VMARG );
    static FALCON_FUNC set_enable_search( VMARG );
    static FALCON_FUNC get_enable_search( VMARG );
    static FALCON_FUNC get_search_column( VMARG );
    static FALCON_FUNC set_search_column( VMARG );
    static FALCON_FUNC get_search_equal_func( VMARG );
    static FALCON_FUNC set_search_equal_func( VMARG );
    static FALCON_FUNC get_search_entry( VMARG );
    static FALCON_FUNC set_search_entry( VMARG );
    static FALCON_FUNC get_search_position_func( VMARG );
    static FALCON_FUNC set_search_position_func( VMARG );
    static FALCON_FUNC get_fixed_height_mode( VMARG );
    static FALCON_FUNC set_fixed_height_mode( VMARG );
    static FALCON_FUNC get_hover_selection( VMARG );
    static FALCON_FUNC set_hover_selection( VMARG );
    static FALCON_FUNC get_hover_expand( VMARG );
    static FALCON_FUNC set_hover_expand( VMARG );
    static FALCON_FUNC set_destroy_count_func( VMARG );
    static FALCON_FUNC get_row_separator_func( VMARG );
    static FALCON_FUNC set_row_separator_func( VMARG );
    static FALCON_FUNC get_rubber_banding( VMARG );
    static FALCON_FUNC set_rubber_banding( VMARG );
    static FALCON_FUNC is_rubber_banding_active( VMARG );
    static FALCON_FUNC get_enable_tree_lines( VMARG );
    static FALCON_FUNC set_enable_tree_lines( VMARG );
    static FALCON_FUNC get_grid_lines( VMARG );
    static FALCON_FUNC set_grid_lines( VMARG );
    static FALCON_FUNC set_tooltip_row( VMARG );
    static FALCON_FUNC set_tooltip_cell( VMARG );
    static FALCON_FUNC get_tooltip_context( VMARG );
    static FALCON_FUNC get_tooltip_column( VMARG );
    static FALCON_FUNC set_tooltip_column( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_TREEVIEW_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
