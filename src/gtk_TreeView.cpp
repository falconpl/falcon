/**
 *  \file gtk_TreeView.cpp
 */

#include "gtk_TreeView.hpp"

#include "gtk_Adjustment.hpp"
#include "gtk_Buildable.hpp"
#include "gtk_CellRenderer.hpp"
//#include "gtk_Extendedlayout.hpp"
#include "gtk_TreeIter.hpp"
#include "gtk_TreeModel.hpp"
#include "gtk_TreePath.hpp"
#include "gtk_TreeViewColumn.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeView::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeView = mod->addClass( "GtkTreeView", &TreeView::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_TreeView->getClassDef()->addInheritance( in );

    c_TreeView->setWKS( true );
    c_TreeView->getClassDef()->factory( &TreeView::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_columns_changed",     &TreeView::signal_columns_changed },
    { "signal_cursor_changed",      &TreeView::signal_cursor_changed },
    { "signal_expand_collapse_cursor_row",&TreeView::signal_expand_collapse_cursor_row },
    { "signal_move_cursor",         &TreeView::signal_move_cursor },
    { "signal_row_activated",       &TreeView::signal_row_activated },
    { "signal_row_collapsed",       &TreeView::signal_row_collapsed },
    { "signal_row_expanded",        &TreeView::signal_row_expanded },
    { "signal_select_all",          &TreeView::signal_select_all },
    { "signal_select_cursor_parent",&TreeView::signal_select_cursor_parent },
    { "signal_select_cursor_row",   &TreeView::signal_select_cursor_row },
    { "signal_set_scroll_adjustments",&TreeView::signal_set_scroll_adjustments },
    { "signal_start_interactive_search",&TreeView::signal_start_interactive_search },
    { "signal_test_collapse_row",   &TreeView::signal_test_collapse_row },
    { "signal_test_expand_row",     &TreeView::signal_test_expand_row },
    { "signal_toggle_cursor_row",   &TreeView::signal_toggle_cursor_row },
    { "signal_unselect_all",        &TreeView::signal_unselect_all },
    { "get_level_indentation",      &TreeView::get_level_indentation },
    { "get_show_expanders",         &TreeView::get_show_expanders },
    { "set_level_indentation",      &TreeView::set_level_indentation },
    { "set_show_expanders",         &TreeView::set_show_expanders },
    { "new_with_model",             &TreeView::new_with_model },
    { "get_model",                  &TreeView::get_model },
    { "set_model",                  &TreeView::set_model },
#if 0 // todo
    { "get_selection",              &TreeView::get_selection },
#endif
    { "get_hadjustment",            &TreeView::get_hadjustment },
    { "set_hadjustment",            &TreeView::set_hadjustment },
    { "get_vadjustment",            &TreeView::get_vadjustment },
    { "set_vadjustment",            &TreeView::set_vadjustment },
    { "get_headers_visible",        &TreeView::get_headers_visible },
    { "set_headers_visible",        &TreeView::set_headers_visible },
    { "columns_autosize",           &TreeView::columns_autosize },
    { "get_headers_clickable",      &TreeView::get_headers_clickable },
    { "set_headers_clickable",      &TreeView::set_headers_clickable },
    { "set_rules_hint",             &TreeView::set_rules_hint },
    { "get_rules_hint",             &TreeView::get_rules_hint },
    { "append_column",              &TreeView::append_column },
    { "remove_column",              &TreeView::remove_column },
    { "insert_column",              &TreeView::insert_column },
    { "insert_column_with_attributes",&TreeView::insert_column_with_attributes },
    { "insert_column_with_data_func",&TreeView::insert_column_with_data_func },
    { "get_column",                 &TreeView::get_column },
    { "get_columns",                &TreeView::get_columns },
    { "move_column_after",          &TreeView::move_column_after },
    { "set_expander_column",        &TreeView::set_expander_column },
    { "get_expander_column",        &TreeView::get_expander_column },
    { "set_column_drag_function",   &TreeView::set_column_drag_function },
    { "scroll_to_point",            &TreeView::scroll_to_point },
    { "scroll_to_cell",             &TreeView::scroll_to_cell },
    { "set_cursor",                 &TreeView::set_cursor },
    { "set_cursor_on_cell",         &TreeView::set_cursor_on_cell },
    { "get_cursor",                 &TreeView::get_cursor },
    { "row_activated",              &TreeView::row_activated },
    { "expand_all",                 &TreeView::expand_all },
    { "collapse_all",               &TreeView::collapse_all },
    { "expand_to_path",             &TreeView::expand_to_path },
    { "expand_row",                 &TreeView::expand_row },
    { "collapse_row",               &TreeView::collapse_row },
#if 0
    { "map_expanded_rows",    &TreeView:: },
    { "row_expanded",    &TreeView:: },
    { "set_reorderable",    &TreeView:: },
    { "get_reorderable",    &TreeView:: },
    { "get_path_at_pos",    &TreeView:: },
    { "get_cell_area",    &TreeView:: },
    { "get_background_area",    &TreeView:: },
    { "get_visible_rect",    &TreeView:: },
    { "get_visible_range",    &TreeView:: },
    { "get_bin_window",    &TreeView:: },
    { "convert_bin_window_to_tree_coords",    &TreeView:: },
    { "convert_bin_window_to_widget_coords",    &TreeView:: },
    { "convert_tree_to_bin_window_coords",    &TreeView:: },
    { "convert_tree_to_widget_coords",    &TreeView:: },
    { "convert_widget_to_bin_window_coords",    &TreeView:: },
    { "convert_widget_to_tree_coords",    &TreeView:: },
    { "enable_model_drag_dest",    &TreeView:: },
    { "enable_model_drag_source",    &TreeView:: },
    { "unset_rows_drag_source",    &TreeView:: },
    { "unset_rows_drag_dest",    &TreeView:: },
    { "set_drag_dest_row",    &TreeView:: },
    { "get_drag_dest_row",    &TreeView:: },
    { "get_dest_row_at_pos",    &TreeView:: },
    { "create_row_drag_icon",    &TreeView:: },
    { "set_enable_search",    &TreeView:: },
    { "get_enable_search",    &TreeView:: },
    { "get_search_column",    &TreeView:: },
    { "set_search_column",    &TreeView:: },
    { "get_search_equal_func",    &TreeView:: },
    { "set_search_equal_func",    &TreeView:: },
    { "get_search_entry",    &TreeView:: },
    { "set_search_entry",    &TreeView:: },
    { "get_search_position_func",    &TreeView:: },
    { "set_search_position_func",    &TreeView:: },
    { "get_fixed_height_mode",    &TreeView:: },
    { "set_fixed_height_mode",    &TreeView:: },
    { "get_hover_selection",    &TreeView:: },
    { "set_hover_selection",    &TreeView:: },
    { "get_hover_expand",    &TreeView:: },
    { "set_hover_expand",    &TreeView:: },
    { "set_destroy_count_func",    &TreeView:: },
    { "get_row_separator_func",    &TreeView:: },
    { "set_row_separator_func",    &TreeView:: },
    { "get_rubber_banding",    &TreeView:: },
    { "set_rubber_banding",    &TreeView:: },
    { "is_rubber_banding_active",    &TreeView:: },
    { "get_enable_tree_lines",    &TreeView:: },
    { "set_enable_tree_lines",    &TreeView:: },
    { "get_grid_lines",    &TreeView:: },
    { "set_grid_lines",    &TreeView:: },
    { "set_tooltip_row",    &TreeView:: },
    { "set_tooltip_cell",    &TreeView:: },
    { "get_tooltip_context",    &TreeView:: },
    { "get_tooltip_column",    &TreeView:: },
    { "set_tooltip_column",    &TreeView:: },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TreeView, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_TreeView );
    //Gtk::ExtendedLayout::clsInit( mod, c_TreeView );
}


TreeView::TreeView( const Falcon::CoreClass* gen, const GtkTreeView* view )
    :
    Gtk::CoreGObject( gen, (GObject*) view )
{}


Falcon::CoreObject* TreeView::factory( const Falcon::CoreClass* gen, void* view, bool )
{
    return new TreeView( gen, (GtkTreeView*) view );
}


/*#
    @class GtkTreeView
    @brief A widget for displaying both trees and lists

    Widget that displays any object that implements the GtkTreeModel interface.

    Please refer to the tree widget conceptual overview for an overview of all
    the objects and data types related to the tree widget and how they work together.

    [...]
 */
FALCON_FUNC TreeView::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_tree_view_new() );
}


/*#
    @method signal_columns_changed
    @brief The number of columns of the treeview has changed.
 */
FALCON_FUNC TreeView::signal_columns_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "columns_changed", (void*) &TreeView::on_columns_changed, vm );
}


void TreeView::on_columns_changed( GtkTreeView* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "columns_changed",
        "on_columns_changed", (VMachine*)_vm );
}


/*#
    @method signal_cursor_changed
    @brief The position of the cursor (focused cell) has changed.
 */
FALCON_FUNC TreeView::signal_cursor_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "cursor_changed", (void*) &TreeView::on_cursor_changed, vm );
}


void TreeView::on_cursor_changed( GtkTreeView* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "cursor_changed",
        "on_cursor_changed", (VMachine*)_vm );
}


/*#
    @method signal_expand_collapse_cursor_row
    @brief The "expand-collapse-cursor-row" signal is emitted when the row at the cursor needs to be expanded or collapsed.
 */
FALCON_FUNC TreeView::signal_expand_collapse_cursor_row( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "expand_collapse_cursor_row", (void*) &TreeView::on_cursor_changed, vm );
}


gboolean TreeView::on_expand_collapse_cursor_row( GtkTreeView* obj, gboolean logical,
                                    gboolean expand, gboolean open_all, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "expand_collapse_cursor_row", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_expand_collapse_cursor_row", it ) )
            {
                printf(
                "[GtkTreeView::on_expand_collapse_cursor_row] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->pushParam( (bool) logical );
        vm->pushParam( (bool) expand );
        vm->pushParam( (bool) open_all );
        vm->callItem( it, 3 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_expand_collapse_cursor_row] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method signal_move_cursor
    @brief The "move-cursor" signal is emitted when the user moves the cursor using the Right, Left, Up or Down arrow keys or the Page Up, Page Down, Home and End keys.
 */
FALCON_FUNC TreeView::signal_move_cursor( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "move_cursor", (void*) &TreeView::on_move_cursor, vm );
}


gboolean TreeView::on_move_cursor( GtkTreeView* obj, GtkMovementStep step,
                                   gint cnt, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "move_cursor", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_move_cursor", it ) )
            {
                printf(
                "[GtkTreeView::on_move_cursor] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->pushParam( (int64) step );
        vm->pushParam( cnt );
        vm->callItem( it, 2 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_move_cursor] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method signal_row_activated
    @brief The "row-activated" signal is emitted when the method gtk_tree_view_row_activated() is called or the user double clicks a treeview row.

    It is also emitted when a non-editable row is selected and one of the keys:
    Space, Shift+Space, Return or Enter is pressed.

    For selection handling refer to the tree widget conceptual overview as well
    as GtkTreeSelection.
 */
FALCON_FUNC TreeView::signal_row_activated( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "row_activated", (void*) &TreeView::on_row_activated, vm );
}


void TreeView::on_row_activated( GtkTreeView* obj, GtkTreePath* path,
                                 GtkTreeViewColumn* column, gpointer _vm )
{ // must free arguments or what?..
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "row_activated", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkTreePath = vm->findWKI( "GtkTreePath" );
    Item* wkTreeViewColumn = vm->findWKI( "GtkTreeViewColumn" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_row_activated", it ) )
            {
                printf(
                "[GtkTreeView::on_row_activated] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreePath( wkTreePath->asClass(), path ) );
        vm->pushParam( new Gtk::TreeViewColumn( wkTreeViewColumn->asClass(), column ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_row_collapsed
    @brief The given row has been collapsed (child nodes are hidden).
 */
FALCON_FUNC TreeView::signal_row_collapsed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "row_collapsed", (void*) &TreeView::on_row_collapsed, vm );
}


void TreeView::on_row_collapsed( GtkTreeView* obj, GtkTreeIter* titer,
                                 GtkTreePath* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "row_collapsed", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkTreeIter = vm->findWKI( "GtkTreeIter" );
    Item* wkTreePath = vm->findWKI( "GtkTreePath" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_row_collapsed", it ) )
            {
                printf(
                "[GtkTreeView::on_row_collapsed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreeIter( wkTreeIter->asClass(), titer ) );
        vm->pushParam( new Gtk::TreePath( wkTreePath->asClass(), path ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_row_expanded
    @brief The given row has been expanded (child nodes are shown).
 */
FALCON_FUNC TreeView::signal_row_expanded( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "row_expanded", (void*) &TreeView::on_row_expanded, vm );
}


void TreeView::on_row_expanded( GtkTreeView* obj, GtkTreeIter* titer,
                                GtkTreePath* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "row_expanded", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkTreeIter = vm->findWKI( "GtkTreeIter" );
    Item* wkTreePath = vm->findWKI( "GtkTreePath" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_row_expanded", it ) )
            {
                printf(
                "[GtkTreeView::on_row_expanded] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreeIter( wkTreeIter->asClass(), titer ) );
        vm->pushParam( new Gtk::TreePath( wkTreePath->asClass(), path ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_select_all
    @brief The "select-all" signal is emitted when the user presses Control+a or Control+/.
 */
FALCON_FUNC TreeView::signal_select_all( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "select_all", (void*) &TreeView::on_select_all, vm );
}


gboolean TreeView::on_select_all( GtkTreeView* obj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "select_all", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_select_all", it ) )
            {
                printf(
                "[GtkTreeView::on_select_all] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_select_all] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method signal_select_cursor_parent
    @brief The "select-cursor-parent" signal is emitted when the user presses Backspace while a row has the cursor.
 */
FALCON_FUNC TreeView::signal_select_cursor_parent( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "select_cursor_parent",
                             (void*) &TreeView::on_select_cursor_parent, vm );
}


gboolean TreeView::on_select_cursor_parent( GtkTreeView* obj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "select_cursor_parent", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_select_cursor_parent", it ) )
            {
                printf(
                "[GtkTreeView::on_select_cursor_parent] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_select_cursor_parent] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method signal_select_cursor_row
    @brief The "select-cursor-row" signal is emitted when a non-editable row is selected and one of the keys: Space, Shift+Space, Return or Enter is pressed.
 */
FALCON_FUNC TreeView::signal_select_cursor_row( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "select_cursor_row",
                             (void*) &TreeView::on_select_cursor_row, vm );
}


gboolean TreeView::on_select_cursor_row( GtkTreeView* obj, gboolean start_editing, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "select_cursor_row", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_select_cursor_row", it ) )
            {
                printf(
                "[GtkTreeView::on_select_cursor_row] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->pushParam( (bool) start_editing );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_select_cursor_row] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method signal_set_scroll_adjustments
    @brief Set Set the scroll adjustments for the tree view.

    Usually scrolled containers like GtkScrolledWindow will emit this signal to
    connect two instances of GtkScrollbar to the scroll directions of the
    GtkTreeView the scroll adjustments for the tree view.
 */
FALCON_FUNC TreeView::signal_set_scroll_adjustments( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "set_scroll_adjustments",
                             (void*) &TreeView::on_set_scroll_adjustments, vm );
}


void TreeView::on_set_scroll_adjustments( GtkTreeView* obj, GtkAdjustment* horiz,
                                          GtkAdjustment* vert, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "set_scroll_adjustments", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkAdjustment" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_set_scroll_adjustments", it ) )
            {
                printf(
                "[GtkTreeView::on_set_scroll_adjustments] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::Adjustment( wki->asClass(), horiz ) );
        vm->pushParam( new Gtk::Adjustment( wki->asClass(), vert ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_start_interactive_search
    @brief The "start-interactive-search" signal is emitted when the user presses Control+f.
 */
FALCON_FUNC TreeView::signal_start_interactive_search( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "start_interactive_search",
                             (void*) &TreeView::on_start_interactive_search, vm );
}


gboolean TreeView::on_start_interactive_search( GtkTreeView* obj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "start_interactive_search", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_start_interactive_search", it ) )
            {
                printf(
                "[GtkTreeView::on_start_interactive_search] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_start_interactive_search] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method signal_test_collapse_row
    @brief The given row is about to be collapsed (hide its children nodes).

    Use this signal if you need to control the collapsibility of individual rows.
 */
FALCON_FUNC TreeView::signal_test_collapse_row( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "test_collapse_row",
                             (void*) &TreeView::on_test_collapse_row, vm );
}


gboolean TreeView::on_test_collapse_row( GtkTreeView* obj, GtkTreeIter* titer,
                                         GtkTreePath* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "test_collapse_row", false );

    if ( !cs || cs->empty() )
        return TRUE; // reject collapsing

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkTreeIter = vm->findWKI( "GtkTreeIter" );
    Item* wkTreePath = vm->findWKI( "GtkTreePath" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_test_collapse_row", it ) )
            {
                printf(
                "[GtkTreeView::on_test_collapse_row] invalid callback (expected callable)\n" );
                return TRUE; // reject collapsing
            }
        }
        vm->pushParam( new Gtk::TreeIter( wkTreeIter->asClass(), titer ) );
        vm->pushParam( new Gtk::TreePath( wkTreePath->asClass(), path ) );
        vm->callItem( it, 2 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( !it.asBoolean() )
                return FALSE; // allow collapsing
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_test_collapse_row] invalid callback (expected boolean)\n" );
            return TRUE; // reject collapsing
        }
    }
    while ( iter.hasCurrent() );

    return TRUE; // reject collapsing
}


/*#
    @method signal_test_expand_row
    @brief The given row is about to be expanded (show its children nodes).

    Use this signal if you need to control the expandability of individual rows.
 */
FALCON_FUNC TreeView::signal_test_expand_row( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "test_expand_row",
                             (void*) &TreeView::on_test_expand_row, vm );
}


gboolean TreeView::on_test_expand_row( GtkTreeView* obj, GtkTreeIter* titer,
                                       GtkTreePath* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "test_expand_row", false );

    if ( !cs || cs->empty() )
        return TRUE; // reject expanding

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkTreeIter = vm->findWKI( "GtkTreeIter" );
    Item* wkTreePath = vm->findWKI( "GtkTreePath" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_test_expand_row", it ) )
            {
                printf(
                "[GtkTreeView::on_test_expand_row] invalid callback (expected callable)\n" );
                return TRUE; // reject expanding
            }
        }
        vm->pushParam( new Gtk::TreeIter( wkTreeIter->asClass(), titer ) );
        vm->pushParam( new Gtk::TreePath( wkTreePath->asClass(), path ) );
        vm->callItem( it, 2 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( !it.asBoolean() )
                return FALSE; // allow expanding
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_test_expand_row] invalid callback (expected boolean)\n" );
            return TRUE; // reject expanding
        }
    }
    while ( iter.hasCurrent() );

    return TRUE; // reject expanding
}


/*#
    @method signal_toggle_cursor_row
    @brief The "toggle-cursor-row" signal is emitted when the user presses Control+Space.
 */
FALCON_FUNC TreeView::signal_toggle_cursor_row( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "toggle_cursor_row",
                             (void*) &TreeView::on_toggle_cursor_row, vm );
}


gboolean TreeView::on_toggle_cursor_row( GtkTreeView* obj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "toggle_cursor_row", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_toggle_cursor_row", it ) )
            {
                printf(
                "[GtkTreeView::on_toggle_cursor_row] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_toggle_cursor_row] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method signal_unselect_all
    @brief The "unselect-all" signal is emitted when the user presses Shift+Control+a or Shift+Control+/.
 */
FALCON_FUNC TreeView::signal_unselect_all( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "unselect_all",
                             (void*) &TreeView::on_unselect_all, vm );
}


gboolean TreeView::on_unselect_all( GtkTreeView* obj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "unselect_all", false );

    if ( !cs || cs->empty() )
        return FALSE; // signal not handled

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_unselect_all", it ) )
            {
                printf(
                "[GtkTreeView::on_unselect_all] invalid callback (expected callable)\n" );
                return FALSE; // signal not handled
            }
        }
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // signal handled
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkTreeView::on_unselect_all] invalid callback (expected boolean)\n" );
            return FALSE; // signal not handled
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // signal not handled
}


/*#
    @method get_level_indentation
    @brief Returns the amount, in pixels, of extra indentation for child levels in tree_view.
    @return the amount of extra indentation for child levels in tree_view. A return value of 0 means that this feature is disabled.
 */
FALCON_FUNC TreeView::get_level_indentation( VMARG )
{
    NO_ARGS
    vm->retval( gtk_tree_view_get_level_indentation( GET_TREEVIEW( vm->self() ) ) );
}


/*#
    @method get_show_expanders
    @brief Returns whether or not expanders are drawn in tree_view.
    @return TRUE if expanders are drawn in tree_view, FALSE otherwise.
 */
FALCON_FUNC TreeView::get_show_expanders( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_get_show_expanders( GET_TREEVIEW( vm->self() ) ) );
}


/*#
    @method set_level_indentation
    @brief Sets the amount of extra indentation for child levels to use in tree_view in addition to the default indentation.
    @param indentation the amount, in pixels, of extra indentation in tree_view.

    The value should be specified in pixels, a value of 0 disables this feature
    and in this case only the default indentation will be used. This does not
    have any visible effects for lists.
 */
FALCON_FUNC TreeView::set_level_indentation( VMARG )
{
    Item* i_ind = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ind || !i_ind->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_tree_view_set_level_indentation( GET_TREEVIEW( vm->self() ),
                                         i_ind->asInteger() );
}


/*#
    @method set_show_expanders
    @brief Sets whether to draw and enable expanders and indent child rows in tree_view.
    @param enabled TRUE to enable expander drawing, FALSE otherwise.

    When disabled there will be no expanders visible in trees and there will be
    no way to expand and collapse rows by default. Also note that hiding the
    expanders will disable the default indentation. You can set a custom
    indentation in this case using gtk_tree_view_set_level_indentation().
    This does not have any visible effects for lists.
 */
FALCON_FUNC TreeView::set_show_expanders( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_set_show_expanders( GET_TREEVIEW( vm->self() ),
                                      (gboolean) i_bool->asBoolean() );
}


/*#
    @method new_with_model
    @brief Creates a new GtkTreeView widget with the model initialized to model.
    @param model the GtkTreeModel
    @return A newly created GtkTreeView widget.
 */
FALCON_FUNC TreeView::new_with_model( VMARG )
{
    Item* i_mdl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mdl || !i_mdl->isObject() || !Gtk::TreeModel::implementedBy( i_mdl ) )
        throw_inv_params( "GtkTreeModel" );
#endif
    GtkTreeView* view = (GtkTreeView*) gtk_tree_view_new_with_model( GET_TREEMODEL( *i_mdl ) );
    vm->retval( new Gtk::TreeView( vm->findWKI( "GtkTreeView" )->asClass(),
                                    view ) );
}


/*#
    @method get_model
    @brief Returns the model the GtkTreeView is based on. Returns NULL if the model is unset.
    @return A GtkTreeModel, or NULL if none is currently being used.
 */
FALCON_FUNC TreeView::get_model( VMARG )
{
    NO_ARGS
    GtkTreeModel* mdl = gtk_tree_view_get_model( GET_TREEVIEW( vm->self() ) );
    if ( mdl )
        vm->retval( new Gtk::TreeModel( vm->findWKI( "GtkTreeModel" )->asClass(), mdl ) );
    else
        vm->retnil();
}


/*#
    @method set_model
    @brief Sets the model for a GtkTreeView.
    @param model The model, or Nil.

    If the tree_view already has a model set, it will remove it before setting
    the new model. If model is NULL, then it will unset the old model.
 */
FALCON_FUNC TreeView::set_model( VMARG )
{
    Item* i_mdl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mdl || !( i_mdl->isNil() || ( i_mdl->isObject()
        && IS_DERIVED( i_mdl, GtkTreeModel ) ) ) )
        throw_inv_params( "[GtkTreeModel]" );
#endif
    GtkTreeModel* mdl = i_mdl->isNil() ? NULL : GET_TREEMODEL( *i_mdl );
    gtk_tree_view_set_model( GET_TREEVIEW( vm->self() ), mdl );
}


#if 0 // todo
FALCON_FUNC TreeView::get_selection( VMARG );
#endif


/*#
    @method get_hadjustment
    @brief Gets the GtkAdjustment currently being used for the horizontal aspect.
    @return A GtkAdjustment object, or NULL if none is currently being used.
 */
FALCON_FUNC TreeView::get_hadjustment( VMARG )
{
    NO_ARGS
    GtkAdjustment* adj = gtk_tree_view_get_hadjustment( GET_TREEVIEW( vm->self() ) );
    if ( adj )
        vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(), adj ) );
    else
        vm->retnil();
}


/*#
    @method set_hadjustment
    @brief Sets the GtkAdjustment for the current horizontal aspect.
    @param adjustment The GtkAdjustment to set, or NULL.
 */
FALCON_FUNC TreeView::set_hadjustment( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) ) )
        throw_inv_params( "[GtkAdjustment]" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL : GET_ADJUSTMENT( *i_adj );
    gtk_tree_view_set_hadjustment( GET_TREEVIEW( vm->self() ), adj );
}


/*#
    @method get_vadjustment
    @brief Gets the GtkAdjustment currently being used for the vertical aspect.
    @return A GtkAdjustment object, or NULL if none is currently being used.
 */
FALCON_FUNC TreeView::get_vadjustment( VMARG )
{
    NO_ARGS
    GtkAdjustment* adj = gtk_tree_view_get_vadjustment( GET_TREEVIEW( vm->self() ) );
    if ( adj )
        vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(), adj ) );
    else
        vm->retnil();
}


/*#
    @method set_vadjustment
    @brief Sets the GtkAdjustment for the current vertical aspect.
    @param adjustment The GtkAdjustment to set, or NULL.
 */
FALCON_FUNC TreeView::set_vadjustment( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) ) )
        throw_inv_params( "[GtkAdjustment]" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL : GET_ADJUSTMENT( *i_adj );
    gtk_tree_view_set_vadjustment( GET_TREEVIEW( vm->self() ), adj );
}


/*#
    @method get_headers_visible
    @brief Returns TRUE if the headers on the tree_view are visible.
    @return Whether the headers are visible or not.
 */
FALCON_FUNC TreeView::get_headers_visible( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_get_headers_visible( GET_TREEVIEW( vm->self() ) ) );
}


/*#
    @method set_headers_visible
    @brief Sets the visibility state of the headers.
    @param headers_visible TRUE if the headers are visible
 */
FALCON_FUNC TreeView::set_headers_visible( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_set_headers_visible( GET_TREEVIEW( vm->self() ),
                                       (gboolean) i_bool->asBoolean() );
}


/*#
    @method columns_autosize
    @brief Resizes all columns to their optimal width.

    Only works after the treeview has been realized.
 */
FALCON_FUNC TreeView::columns_autosize( VMARG )
{
    NO_ARGS
    gtk_tree_view_columns_autosize( GET_TREEVIEW( vm->self() ) );
}


/*#
    @method get_headers_clickable
    @brief Returns whether all header columns are clickable.
    @return TRUE if all header columns are clickable, otherwise FALSE
 */
FALCON_FUNC TreeView::get_headers_clickable( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_get_headers_clickable( GET_TREEVIEW( vm->self() ) ) );
}


/*#
    @method set_headers_clickable
    @brief Allow the column title buttons to be clicked.
    @param setting TRUE if the columns are clickable.
 */
FALCON_FUNC TreeView::set_headers_clickable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_set_headers_clickable( GET_TREEVIEW( vm->self() ),
                                         (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_rules_hint
    @brief This function tells GTK+ that the user interface for your application requires users to read across tree rows and associate cells with one another.
    @param setting TRUE if the tree requires reading across rows

    By default, GTK+ will then render the tree with alternating row colors.
    Do not use it just because you prefer the appearance of the ruled tree;
    that's a question for the theme. Some themes will draw tree rows in
    alternating colors even when rules are turned off, and users who prefer that
    appearance all the time can choose those themes. You should call this
    function only as a semantic hint to the theme engine that your tree makes
    alternating colors useful from a functional standpoint (since it has lots of
    columns, generally).
 */
FALCON_FUNC TreeView::set_rules_hint( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_set_rules_hint( GET_TREEVIEW( vm->self() ),
                                  (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_rules_hint
    @brief Gets the setting set by gtk_tree_view_set_rules_hint().
    @return TRUE if rules are useful for the user of this tree
 */
FALCON_FUNC TreeView::get_rules_hint( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_get_rules_hint( GET_TREEVIEW( vm->self() ) ) );
}


/*#
    @method append_column
    @brief Appends column to the list of columns.
    @param column The GtkTreeViewColumn to add.
    @return The number of columns in tree_view after appending.

    If tree_view has "fixed_height" mode enabled, then column must have its
    "sizing" property set to be GTK_TREE_VIEW_COLUMN_FIXED.
 */
FALCON_FUNC TreeView::append_column( VMARG )
{
    Item* i_col = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || !i_col->isObject() || !IS_DERIVED( i_col, GtkTreeViewColumn ) )
        throw_inv_params( "GtkTreeViewColumn" );
#endif
    GtkTreeViewColumn* col = GET_TREEVIEWCOLUMN( *i_col );
    vm->retval( gtk_tree_view_append_column( GET_TREEVIEW( vm->self() ), col ) );
}


/*#
    @method remove_column
    @brief Removes column from tree_view.
    @param column The GtkTreeViewColumn to remove.
    @return The number of columns in tree_view after removing.
 */
FALCON_FUNC TreeView::remove_column( VMARG )
{
    Item* i_col = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || !i_col->isObject() || !IS_DERIVED( i_col, GtkTreeViewColumn ) )
        throw_inv_params( "GtkTreeViewColumn" );
#endif
    GtkTreeViewColumn* col = GET_TREEVIEWCOLUMN( *i_col );
    vm->retval( gtk_tree_view_remove_column( GET_TREEVIEW( vm->self() ), col ) );
}


/*#
    @method insert_column
    @brief This inserts the column into the tree_view at position.
    @param column The GtkTreeViewColumn to be inserted.
    @param position The position to insert column in.
    @return The number of columns in tree_view after insertion.

    If position is -1, then the column is inserted at the end. If tree_view has
    "fixed_height" mode enabled, then column must have its "sizing" property set
    to be GTK_TREE_VIEW_COLUMN_FIXED.
 */
FALCON_FUNC TreeView::insert_column( VMARG )
{
    Item* i_col = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || !i_col->isObject() || !IS_DERIVED( i_col, GtkTreeViewColumn )
        || !i_pos || i_pos->isInteger() )
        throw_inv_params( "GtkTreeViewColumn,I" );
#endif
    GtkTreeViewColumn* col = GET_TREEVIEWCOLUMN( *i_col );
    vm->retval( gtk_tree_view_insert_column( GET_TREEVIEW( vm->self() ),
                                             col, i_pos->asInteger() ) );
}


/*#
    @method insert_column_with_attributes
    @brief Creates a new GtkTreeViewColumn and inserts it into the tree_view at position.
    @param position The position to insert the new column in.
    @param title The title to set the header to.
    @param cell The GtkCellRenderer.
    @param attributes An array of pairs [ attribute, column, ... ]
    @return The number of columns in tree_view after insertion.

    If position is -1, then the newly created column is inserted at the end. The
    column is initialized with the attributes given. If tree_view has
    "fixed_height" mode enabled, then the new column will have its sizing
    property set to be GTK_TREE_VIEW_COLUMN_FIXED.
 */
FALCON_FUNC TreeView::insert_column_with_attributes( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_title = vm->param( 1 );
    Item* i_cell = vm->param( 2 );
    Item* i_attr = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_title || !i_title->isString()
        || !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_attr || !i_attr->isArray() )
        throw_inv_params( "I,S,GtkCellRenderer,A" );
#endif
    AutoCString title( i_title->asString() );
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    CoreArray* attr = i_attr->asArray();
    const int len = attr->length();
#ifndef NO_PARAMETER_CHECK
    if ( len == 0 )
        throw_inv_params( "Non-empty array" );
    if ( len % 2 != 0 )
        throw_inv_params( "Array of pairs" );
#endif
    GtkTreeViewColumn* col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title( col, title.c_str() );
    Item it;
    for ( int i = 0; i < len; i += 2 )
    {
        it = attr->at( i );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isString() )
        {
            g_object_unref( col );
            throw_inv_params( "S" );
        }
#endif
        AutoCString key( it.asString() );
        it = attr->at( i + 1 );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isInteger() )
        {
            g_object_unref( col );
            throw_inv_params( "I" );
        }
#endif
        gtk_tree_view_column_add_attribute( col,
                                            cell, key.c_str(), it.asInteger() );
    }
    vm->retval( gtk_tree_view_insert_column( GET_TREEVIEW( vm->self() ),
                                                    col, i_pos->asInteger() ) );
}


/*#
    @method insert_column_with_data_func
    @brief Convenience function that inserts a new column into the GtkTreeView with the given cell renderer and a GtkCellDataFunc to set cell renderer attributes (normally using data from the model).
    @param position Position to insert, -1 for append
    @param title column title
    @param cell cell renderer for column
    @param func function to set attributes of cell renderer, or nil
    @param data data for func, or nil

    See also gtk_tree_view_column_set_cell_data_func(),
    gtk_tree_view_column_pack_start(). If tree_view has "fixed_height" mode enabled,
    then the new column will have its "sizing" property set to be
    GTK_TREE_VIEW_COLUMN_FIXED.
 */
FALCON_FUNC TreeView::insert_column_with_data_func( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_title = vm->param( 1 );
    Item* i_cell = vm->param( 2 );
    Item* i_func = vm->param( 3 );
    Item* i_data = vm->param( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_title || !i_title->isString()
        || !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_func || !( i_func->isNil() || i_func->isCallable() )
        || !i_data )
        throw_inv_params( "I,GtkCellRenderer,[C],[X]" );
#endif
    AutoCString title( i_title->asString() );
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    GtkTreeViewColumn* col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title( col, title.c_str() );
    if ( !i_func->isNil() )
    {
        g_object_set_data_full( (GObject*) col,
                                "__tree_view_column_cell_data_func__",
                                new GarbageLock( *i_func ),
                                &CoreGObject::release_lock );
        g_object_set_data_full( (GObject*) col,
                                "__tree_view_column_cell_data_func_data__",
                                new GarbageLock( *i_data ),
                                &CoreGObject::release_lock );
        gtk_tree_view_column_set_cell_data_func( col,
                                                 cell,
                                                 &TreeViewColumn::exec_cell_data_func,
                                                 (gpointer) vm,
                                                 NULL );
    }
    vm->retval( gtk_tree_view_insert_column( GET_TREEVIEW( vm->self() ),
                                             col, i_pos->asInteger() ) );
}


/*#
    @method get_column
    @brief Gets the GtkTreeViewColumn at the given position in the tree_view.
    @param n The position of the column, counting from 0.
    @return The GtkTreeViewColumn, or NULL if the position is outside the range of columns.
 */
FALCON_FUNC TreeView::get_column( VMARG )
{
    Item* i_n = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_n || !i_n->isInteger() )
        throw_inv_params( "I" );
#endif
    GtkTreeViewColumn* col = gtk_tree_view_get_column( GET_TREEVIEW( vm->self() ),
                                                       i_n->asInteger() );
    if ( col )
        vm->retval( new Gtk::TreeViewColumn( vm->findWKI( "GtkTreeViewColumn" )->asClass(),
                                             col ) );
    else
        vm->retnil();
}


/*#
    @method get_columns
    @brief Returns an array of all the GtkTreeViewColumn currently in tree_view.
    @return A list of GtkTreeViewColumn
 */
FALCON_FUNC TreeView::get_columns( VMARG )
{
    NO_ARGS
    GList* lst = gtk_tree_view_get_columns( GET_TREEVIEW( vm->self() ) );
    GList* el;
    int cnt = 0;
    for ( el = lst; el; el = el->next ) ++cnt;
    CoreArray* arr = new CoreArray( cnt );
    if ( cnt )
    {
        Item* wki = vm->findWKI( "GtkTreeViewColumn" );
        for ( el = lst; el; el = el->next )
            arr->append( new Gtk::TreeViewColumn( wki->asClass(),
                                                  (GtkTreeViewColumn*) el->data ) );
    }
    vm->retval( arr );
}


/*#
    @method move_column_after
    @brief Moves column to be after to base_column.
    @param column The GtkTreeViewColumn to be moved.
    @param base_column The GtkTreeViewColumn to be moved relative to, or NULL.

    If base_column is NULL, then column is placed in the first position.
 */
FALCON_FUNC TreeView::move_column_after( VMARG )
{
    Item* i_col = vm->param( 0 );
    Item* i_base = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || !i_col->isObject() || !IS_DERIVED( i_col, GtkTreeViewColumn )
        || !i_base || !( i_base->isNil() || ( i_base->isObject()
        && IS_DERIVED( i_base, GtkTreeViewColumn ) ) ) )
        throw_inv_params( "GtkTreeViewColumn,[GtkTreeViewColumn]" );
#endif
    GtkTreeViewColumn* col = GET_TREEVIEWCOLUMN( *i_col );
    GtkTreeViewColumn* base = i_base->isNil() ? NULL : GET_TREEVIEWCOLUMN( *i_base );
    gtk_tree_view_move_column_after( GET_TREEVIEW( vm->self() ), col, base );
}


/*#
    @method set_expander_column
    @brief Sets the column to draw the expander arrow at.
    @param column NULL, or the GtkTreeViewColumn to draw the expander arrow at.

    It must be in tree_view. If column is NULL, then the expander arrow is always
    at the first visible column.

    If you do not want expander arrow to appear in your tree, set the expander
    column to a hidden column.
 */
FALCON_FUNC TreeView::set_expander_column( VMARG )
{
    Item* i_col = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || !( i_col->isNil() || ( i_col->isObject()
        && IS_DERIVED( i_col, GtkTreeViewColumn ) ) ) )
        throw_inv_params( "[GtkTreeViewColumn]" );
#endif
    GtkTreeViewColumn* col = i_col->isNil() ? NULL : GET_TREEVIEWCOLUMN( *i_col );
    gtk_tree_view_set_expander_column( GET_TREEVIEW( vm->self() ), col );
}


/*#
    @method get_expander_column
    @brief Returns the column that is the current expander column.
    @return The expander column.

    This column has the expander arrow drawn next to it.
 */
FALCON_FUNC TreeView::get_expander_column( VMARG )
{
    NO_ARGS
    GtkTreeViewColumn* col = gtk_tree_view_get_expander_column( GET_TREEVIEW( vm->self() ) );
    vm->retval( new Gtk::TreeViewColumn( vm->findWKI( "GtkTreeViewColumn" )->asClass(),
                                         col ) );
}


/*#
    @method set_column_drag_function
    @brief Sets a user function for determining where a column may be dropped when dragged.
    @param func A function to determine which columns are reorderable, or NULL.
    @param user_data User data to be passed to func, or NULL.

    This function is called on every column pair in turn at the beginning of a
    column drag to determine where a drop can take place. The arguments passed
    to func are: the GtkTreeViewColumn being dragged, the two GtkTreeViewColumn
    determining the drop spot, and user_data. If either of the GtkTreeViewColumn
    arguments for the drop spot are NULL, then they indicate an edge. If func is
    set to be NULL, then tree_view reverts to the default behavior of allowing
    all columns to be dropped everywhere.
 */
FALCON_FUNC TreeView::set_column_drag_function( VMARG )
{
    Item* i_func = vm->param( 0 );
    Item* i_data = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_func || !( i_func->isNil() || i_func->isCallable() )
        || !i_data )
        throw_inv_params( "[C],[X]" );
#endif
    MYSELF;
    GET_OBJ( self );
    if ( i_func->isNil() )
    {
        if ( g_object_get_data( (GObject*)_obj,
                                "__tree_view_column_drag_func__" ) )
        {
            g_object_set_data( (GObject*)_obj,
                               "__tree_view_column_drag_func__", NULL );
            g_object_set_data( (GObject*)_obj,
                               "__tree_view_column_drag_func_data__", NULL );
        }
        gtk_tree_view_set_column_drag_function( (GtkTreeView*)_obj,
                                                NULL, NULL, NULL );
    }
    else
    {
        g_object_set_data_full( (GObject*)_obj,
                                "__tree_view_column_drag_func__",
                                new GarbageLock( *i_func ),
                                &CoreGObject::release_lock );
        g_object_set_data_full( (GObject*)_obj,
                                "__tree_view_column_drag_func_data__",
                                new GarbageLock( *i_data ),
                                &CoreGObject::release_lock );
        gtk_tree_view_set_column_drag_function( (GtkTreeView*)_obj,
                                                &TreeView::exec_column_drag_function,
                                                (gpointer) vm,
                                                NULL );
    }
}


gboolean TreeView::exec_column_drag_function( GtkTreeView* obj,
                                              GtkTreeViewColumn* col,
                                              GtkTreeViewColumn* prev_col,
                                              GtkTreeViewColumn* next_col,
                                              gpointer _vm )
{
    GarbageLock* func_lock = (GarbageLock*) g_object_get_data( (GObject*) obj,
                                            "__tree_view_column_drag_func__" );
    GarbageLock* data_lock = (GarbageLock*) g_object_get_data( (GObject*) obj,
                                            "__tree_view_column_drag_func_data__" );
    assert( func_lock && data_lock );
    Item func = func_lock->item();
    Item data = func_lock->item();
    VMachine* vm = (VMachine*) _vm;
    Item* wki = vm->findWKI( "GtkTreeViewColumn" );
    vm->pushParam( col ? new Gtk::TreeViewColumn( wki->asClass(), col ) : Item( FLC_ITEM_NIL ) );
    vm->pushParam( prev_col ? new Gtk::TreeViewColumn( wki->asClass(), prev_col ) : Item( FLC_ITEM_NIL ) );
    vm->pushParam( next_col ? new Gtk::TreeViewColumn( wki->asClass(), next_col ) : Item( FLC_ITEM_NIL ) );
    vm->pushParam( data );
    vm->callItem( func, 4 );
    Item it = vm->regA();
    if ( !it.isBoolean() )
    {
        g_print( "TreeView::exec_column_drag_function: invalid return value (expected boolean)\n" );
        return FALSE;
    }
    else
        return (gboolean) it.asBoolean();
}


/*#
    @method scroll_to_point
    @brief Scrolls the tree view such that the top-left corner of the visible area is tree_x, tree_y, where tree_x and tree_y are specified in tree coordinates.
    @param tree_x X coordinate of new top-left pixel of visible area, or -1
    @param tree_y Y coordinate of new top-left pixel of visible area, or -1

    The tree_view must be realized before this function is called. If it isn't,
    you probably want to be using gtk_tree_view_scroll_to_cell().

    If either tree_x or tree_y are -1, then that direction isn't scrolled.
 */
FALCON_FUNC TreeView::scroll_to_point( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    gtk_tree_view_scroll_to_point( GET_TREEVIEW( vm->self() ),
                                   i_x->asInteger(),
                                   i_y->asInteger() );
}


/*#
    @method scroll_to_cell
    @brief Moves the alignments of tree_view to the position specified by column and path.
    @param path The GtkTreePath of the row to move to, or NULL.
    @param column The GtkTreeViewColumn to move horizontally to, or NULL.
    @param use_align whether to use alignment arguments, or FALSE.
    @param row_align The vertical alignment (numeric) of the row specified by path.
    @param col_align The horizontal alignment (numeric) of the column specified by column.

    If column is NULL, then no horizontal scrolling occurs. Likewise, if path is
    NULL no vertical scrolling occurs. At a minimum, one of column or path need
    to be non-NULL. row_align determines where the row is placed, and col_align determines where column is placed. Both are expected to be between 0.0 and 1.0. 0.0 means left/top alignment, 1.0 means right/bottom alignment, 0.5 means center.

    If use_align is FALSE, then the alignment arguments are ignored, and the
    tree does the minimum amount of work to scroll the cell onto the screen.
    This means that the cell will be scrolled to the edge closest to its current
    position. If the cell is currently visible on the screen, nothing is done.

    This function only works if the model is set, and path is a valid row on the
    model. If the model changes before the tree_view is realized, the centered
    path will be modified to reflect this change.
 */
FALCON_FUNC TreeView::scroll_to_cell( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_col = vm->param( 1 );
    Item* i_use_align = vm->param( 2 );
    Item* i_row_align = vm->param( 3 );
    Item* i_col_align = vm->param( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !( i_path->isNil() || ( i_path->isObject()
        && IS_DERIVED( i_path, GtkTreePath ) ) )
        || !i_col || !( i_col->isNil() || ( i_col->isObject()
        && IS_DERIVED( i_col, GtkTreeViewColumn ) ) )
        || !i_use_align || !i_use_align->isBoolean()
        || !i_row_align || !i_row_align->isOrdinal()
        || !i_col_align || !i_col_align->isOrdinal() )
        throw_inv_params( "[GtkTreePath],[GtkTreeViewColumn],B,N,N" );
#endif
    GtkTreePath* path = i_path->isNil() ? NULL : GET_TREEPATH( *i_path );
    GtkTreeViewColumn* col = i_col->isNil() ? NULL : GET_TREEVIEWCOLUMN( *i_col );
    gtk_tree_view_scroll_to_cell( GET_TREEVIEW( vm->self() ),
                                  path,
                                  col,
                                  (gboolean) i_use_align->asBoolean(),
                                  i_row_align->forceNumeric(),
                                  i_col_align->forceNumeric() );
}


/*#
    @method set_cursor
    @brief Sets the current keyboard focus to be at path, and selects it.
    @param path A GtkTreePath
    @param focus_column A GtkTreeViewColumn, or NULL.
    @param start_editing TRUE if the specified cell should start being edited.

    This is useful when you want to focus the user's attention on a particular
    row. If focus_column is not NULL, then focus is given to the column specified
    by it. Additionally, if focus_column is specified, and start_editing is TRUE,
    then editing should be started in the specified cell. This function is often
    followed by gtk_widget_grab_focus (tree_view) in order to give keyboard focus
    to the widget. Please note that editing can only happen when the widget is realized.

    If path is invalid for model, the current cursor (if any) will be unset and
    the function will return without failing.
 */
FALCON_FUNC TreeView::set_cursor( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_col = vm->param( 1 );
    Item* i_edit = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_col || !( i_col->isNil() || ( i_col->isObject()
        && IS_DERIVED( i_col, GtkTreeViewColumn ) ) )
        || !i_edit || !i_edit->isBoolean() )
        throw_inv_params( "GtkTreePath,[GtkTreeViewColumn],B" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeViewColumn* col = i_col->isNil() ? NULL : GET_TREEVIEWCOLUMN( *i_col );
    gtk_tree_view_set_cursor( GET_TREEVIEW( vm->self() ),
                              path, col, (gboolean) i_edit->asBoolean() );
}


/*#
    @method set_cursor_on_cell
    @brief Sets the current keyboard focus to be at path, and selects it.
    @param path A GtkTreePath
    @param focus_column A GtkTreeViewColumn, or NULL.
    @param focus_cell A GtkCellRenderer, or NULL.
    @param start_editing TRUE if the specified cell should start being edited.

    This is useful when you want to focus the user's attention on a particular
    row. If focus_column is not NULL, then focus is given to the column specified
    by it. If focus_column and focus_cell are not NULL, and focus_column contains
    2 or more editable or activatable cells, then focus is given to the cell
    specified by focus_cell. Additionally, if focus_column is specified, and
    start_editing is TRUE, then editing should be started in the specified cell.
    This function is often followed by gtk_widget_grab_focus (tree_view) in
    order to give keyboard focus to the widget. Please note that editing can
    only happen when the widget is realized.

    If path is invalid for model, the current cursor (if any) will be unset and
    the function will return without failing.
 */
FALCON_FUNC TreeView::set_cursor_on_cell( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_col = vm->param( 1 );
    Item* i_cell = vm->param( 2 );
    Item* i_edit = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_col || !( i_col->isNil() || ( i_col->isObject()
        && IS_DERIVED( i_col, GtkTreeViewColumn ) ) )
        || !i_cell || !( i_cell->isNil() || ( i_cell->isObject()
        && IS_DERIVED( i_cell, GtkCellRenderer ) ) )
        || !i_edit || !i_edit->isBoolean() )
        throw_inv_params( "GtkTreePath,[GtkTreeViewColumn],[GtkCellRenderer],B" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeViewColumn* col = i_col->isNil() ? NULL : GET_TREEVIEWCOLUMN( *i_col );
    GtkCellRenderer* cell = i_cell->isNil() ? NULL : GET_CELLRENDERER( *i_cell );
    gtk_tree_view_set_cursor_on_cell( GET_TREEVIEW( vm->self() ),
                              path, col, cell, (gboolean) i_edit->asBoolean() );
}


/*#
    @method get_cursor
    @brief Fills in path and focus_column with the current path and focus column.
    @return An array [ current cursor path, current focus column ]

    If the cursor isn't currently set, then the returned path will be NULL.
    If no column currently has focus, then the returned focus_column will be NULL.
 */
FALCON_FUNC TreeView::get_cursor( VMARG )
{
    NO_ARGS
    GtkTreePath* path;
    GtkTreeViewColumn* col;
    gtk_tree_view_get_cursor( GET_TREEVIEW( vm->self() ), &path, &col );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( path ? new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                           path, true ) : Item( FLC_ITEM_NIL ) );
    arr->append( col ? new Gtk::TreeViewColumn( vm->findWKI( "GtkTreeViewColumn" )->asClass(),
                                                col ) : Item( FLC_ITEM_NIL ) );
    vm->retval( arr );
}


/*#
    @method row_activated
    @brief Activates the cell determined by path and column.
    @param path The GtkTreePath to be activated.
    @param column The GtkTreeViewColumn to be activated.
 */
FALCON_FUNC TreeView::row_activated( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_col = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_col || !i_col->isObject() || !IS_DERIVED( i_col, GtkTreeViewColumn ) )
        throw_inv_params( "GtkTreePath,GtkTreeViewColumn" );
#endif
    gtk_tree_view_row_activated( GET_TREEVIEW( vm->self() ),
                                 GET_TREEPATH( *i_path ),
                                 GET_TREEVIEWCOLUMN( *i_col ) );
}


/*#
    @method expand_all
    @brief Recursively expands all nodes in the tree_view.
 */
FALCON_FUNC TreeView::expand_all( VMARG )
{
    NO_ARGS
    gtk_tree_view_expand_all( GET_TREEVIEW( vm->self() ) );
}


/*#
    @method collapse_all
    @brief Recursively collapses all visible, expanded nodes in tree_view.
 */
FALCON_FUNC TreeView::collapse_all( VMARG )
{
    NO_ARGS
    gtk_tree_view_collapse_all( GET_TREEVIEW( vm->self() ) );
}


/*#
    @method expand_to_path
    @brief Expands the row at path.
    @param path GtkTreePath to a row.

    This will also expand all parent rows of path as necessary.
 */
FALCON_FUNC TreeView::expand_to_path( VMARG )
{
    Item* i_path = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    gtk_tree_view_expand_to_path( GET_TREEVIEW( vm->self() ),
                                  GET_TREEPATH( *i_path ) );
}


/*#
    @method expand_row
    @brief Opens the row so its children are visible.
    @param path GtkTreePath to a row
    @param open_all whether to recursively expand, or just expand immediate children
    @return TRUE if the row existed and had children
 */
FALCON_FUNC TreeView::expand_row( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_open = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_open || !i_open->isBoolean() )
        throw_inv_params( "GtkTreePath,B" );
#endif
    vm->retval( (bool) gtk_tree_view_expand_row( GET_TREEVIEW( vm->self() ),
                                                 GET_TREEPATH( *i_path ),
                                                 (gboolean) i_open->asBoolean() ) );
}


/*#
    @method collapse_row
    @brief Collapses a row (hides its child rows, if they exist).
    @param path path to a row in the tree_view.
    @return TRUE if the row was collapsed.
 */
FALCON_FUNC TreeView::collapse_row( VMARG )
{
    Item* i_path = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    vm->retval( (bool) gtk_tree_view_collapse_row( GET_TREEVIEW( vm->self() ),
                                                   GET_TREEPATH( *i_path ) ) );
}


#if 0
FALCON_FUNC TreeView::map_expanded_rows( VMARG );
FALCON_FUNC TreeView::row_expanded( VMARG );
FALCON_FUNC TreeView::set_reorderable( VMARG );
FALCON_FUNC TreeView::get_reorderable( VMARG );
FALCON_FUNC TreeView::get_path_at_pos( VMARG );
FALCON_FUNC TreeView::get_cell_area( VMARG );
FALCON_FUNC TreeView::get_background_area( VMARG );
FALCON_FUNC TreeView::get_visible_rect( VMARG );
FALCON_FUNC TreeView::get_visible_range( VMARG );
FALCON_FUNC TreeView::get_bin_window( VMARG );
FALCON_FUNC TreeView::convert_bin_window_to_tree_coords( VMARG );
FALCON_FUNC TreeView::convert_bin_window_to_widget_coords( VMARG );
FALCON_FUNC TreeView::convert_tree_to_bin_window_coords( VMARG );
FALCON_FUNC TreeView::convert_tree_to_widget_coords( VMARG );
FALCON_FUNC TreeView::convert_widget_to_bin_window_coords( VMARG );
FALCON_FUNC TreeView::convert_widget_to_tree_coords( VMARG );
FALCON_FUNC TreeView::enable_model_drag_dest( VMARG );
FALCON_FUNC TreeView::enable_model_drag_source( VMARG );
FALCON_FUNC TreeView::unset_rows_drag_source( VMARG );
FALCON_FUNC TreeView::unset_rows_drag_dest( VMARG );
FALCON_FUNC TreeView::set_drag_dest_row( VMARG );
FALCON_FUNC TreeView::get_drag_dest_row( VMARG );
FALCON_FUNC TreeView::get_dest_row_at_pos( VMARG );
FALCON_FUNC TreeView::create_row_drag_icon( VMARG );
FALCON_FUNC TreeView::set_enable_search( VMARG );
FALCON_FUNC TreeView::get_enable_search( VMARG );
FALCON_FUNC TreeView::get_search_column( VMARG );
FALCON_FUNC TreeView::set_search_column( VMARG );
FALCON_FUNC TreeView::get_search_equal_func( VMARG );
FALCON_FUNC TreeView::set_search_equal_func( VMARG );
FALCON_FUNC TreeView::get_search_entry( VMARG );
FALCON_FUNC TreeView::set_search_entry( VMARG );
FALCON_FUNC TreeView::get_search_position_func( VMARG );
FALCON_FUNC TreeView::set_search_position_func( VMARG );
FALCON_FUNC TreeView::get_fixed_height_mode( VMARG );
FALCON_FUNC TreeView::set_fixed_height_mode( VMARG );
FALCON_FUNC TreeView::get_hover_selection( VMARG );
FALCON_FUNC TreeView::set_hover_selection( VMARG );
FALCON_FUNC TreeView::get_hover_expand( VMARG );
FALCON_FUNC TreeView::set_hover_expand( VMARG );
FALCON_FUNC TreeView::set_destroy_count_func( VMARG );
FALCON_FUNC TreeView::get_row_separator_func( VMARG );
FALCON_FUNC TreeView::set_row_separator_func( VMARG );
FALCON_FUNC TreeView::get_rubber_banding( VMARG );
FALCON_FUNC TreeView::set_rubber_banding( VMARG );
FALCON_FUNC TreeView::is_rubber_banding_active( VMARG );
FALCON_FUNC TreeView::get_enable_tree_lines( VMARG );
FALCON_FUNC TreeView::set_enable_tree_lines( VMARG );
FALCON_FUNC TreeView::get_grid_lines( VMARG );
FALCON_FUNC TreeView::set_grid_lines( VMARG );
FALCON_FUNC TreeView::set_tooltip_row( VMARG );
FALCON_FUNC TreeView::set_tooltip_cell( VMARG );
FALCON_FUNC TreeView::get_tooltip_context( VMARG );
FALCON_FUNC TreeView::get_tooltip_column( VMARG );
FALCON_FUNC TreeView::set_tooltip_column( VMARG );
#endif

} // Gtk
} // Falcon
