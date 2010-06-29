/**
 *  \file gtk_TreeViewColumn.cpp
 */

#include "gtk_TreeViewColumn.hpp"

#include "gdk_Rectangle.hpp"

#include "gtk_Buildable.hpp"
//#include "gtk_CellLayout.hpp"
#include "gtk_CellRenderer.hpp"
#include "gtk_TreeIter.hpp"
#include "gtk_TreeModel.hpp"
#include "gtk_TreeView.hpp"
#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeViewColumn::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeViewColumn = mod->addClass( "GtkTreeViewColumn", &TreeViewColumn::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkObject" ) );
    c_TreeViewColumn->getClassDef()->addInheritance( in );

    c_TreeViewColumn->setWKS( true );
    c_TreeViewColumn->getClassDef()->factory( &TreeViewColumn::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_clicked",         &TreeViewColumn::signal_clicked },
    { "new_with_attributes",    &TreeViewColumn::new_with_attributes },
    { "pack_start",             &TreeViewColumn::pack_start },
    { "pack_end",               &TreeViewColumn::pack_end },
    { "clear",                  &TreeViewColumn::clear },
    { "add_attribute",          &TreeViewColumn::add_attribute },
    { "set_attributes",         &TreeViewColumn::set_attributes },
    { "set_cell_data_func",     &TreeViewColumn::set_cell_data_func },
    { "clear_attributes",       &TreeViewColumn::clear_attributes },
    { "set_spacing",            &TreeViewColumn::set_spacing },
    { "get_spacing",            &TreeViewColumn::get_spacing },
    { "set_visible",            &TreeViewColumn::set_visible },
    { "get_visible",            &TreeViewColumn::get_visible },
    { "set_resizable",          &TreeViewColumn::set_resizable },
    { "get_resizable",          &TreeViewColumn::get_resizable },
    { "set_sizing",             &TreeViewColumn::set_sizing },
    { "get_sizing",             &TreeViewColumn::get_sizing },
    { "get_width",              &TreeViewColumn::get_width },
    { "get_fixed_width",        &TreeViewColumn::get_fixed_width },
    { "set_fixed_width",        &TreeViewColumn::set_fixed_width },
    { "set_min_width",          &TreeViewColumn::set_min_width },
    { "get_min_width",          &TreeViewColumn::get_min_width },
    { "set_max_width",          &TreeViewColumn::set_max_width },
    { "get_max_width",          &TreeViewColumn::get_max_width },
    { "clicked",                &TreeViewColumn::clicked },
    { "set_title",              &TreeViewColumn::set_title },
    { "get_title",              &TreeViewColumn::get_title },
    { "set_expand",             &TreeViewColumn::set_expand },
    { "get_expand",             &TreeViewColumn::get_expand },
    { "set_clickable",          &TreeViewColumn::set_clickable },
    { "get_clickable",          &TreeViewColumn::get_clickable },
    { "set_widget",             &TreeViewColumn::set_widget },
    { "get_widget",             &TreeViewColumn::get_widget },
    { "set_alignment",          &TreeViewColumn::set_alignment },
    { "get_alignment",          &TreeViewColumn::get_alignment },
    { "set_reorderable",        &TreeViewColumn::set_reorderable },
    { "get_reorderable",        &TreeViewColumn::get_reorderable },
    { "set_sort_column_id",     &TreeViewColumn::set_sort_column_id },
    { "get_sort_column_id",     &TreeViewColumn::get_sort_column_id },
    { "set_sort_indicator",     &TreeViewColumn::set_sort_indicator },
    { "get_sort_indicator",     &TreeViewColumn::get_sort_indicator },
    { "set_sort_order",         &TreeViewColumn::set_sort_order },
    { "get_sort_order",         &TreeViewColumn::get_sort_order },
    { "cell_set_cell_data",     &TreeViewColumn::cell_set_cell_data },
    { "cell_get_size",          &TreeViewColumn::cell_get_size },
    { "cell_get_position",      &TreeViewColumn::cell_get_position },
    { "cell_is_visible",        &TreeViewColumn::cell_is_visible },
    { "focus_cell",             &TreeViewColumn::focus_cell },
    { "queue_resize",           &TreeViewColumn::queue_resize },
    { "get_tree_view",          &TreeViewColumn::get_tree_view },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TreeViewColumn, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_TreeViewColumn );
    //Gtk::CellLayout::clsInit( mod, c_TreeViewColumn );
}


TreeViewColumn::TreeViewColumn( const Falcon::CoreClass* gen, const GtkTreeViewColumn* view )
    :
    Gtk::CoreGObject( gen, (GObject*) view )
{}


Falcon::CoreObject* TreeViewColumn::factory( const Falcon::CoreClass* gen, void* view, bool )
{
    return new TreeViewColumn( gen, (GtkTreeViewColumn*) view );
}


/*#
    @class GtkTreeViewColumn
    @brief A visible column in a GtkTreeView widget

    The GtkTreeViewColumn object represents a visible column in a GtkTreeView
    widget. It allows to set properties of the column header, and functions as
    a holding pen for the cell renderers which determine how the data in the
    column is displayed.

    Please refer to the tree widget conceptual overview for an overview of all
    the objects and data types related to the tree widget and how they work together.
 */
FALCON_FUNC TreeViewColumn::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_tree_view_column_new() );
}


/*#
    @method signal_clicked GtkTreeViewColumn
    @brief ?
 */
FALCON_FUNC TreeViewColumn::signal_clicked( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "clicked", (void*) &TreeViewColumn::on_clicked, vm );
}


void TreeViewColumn::on_clicked( GtkTreeViewColumn* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "clicked", "on_clicked", (VMachine*)_vm );
}


/*#
    @method new_with_attributes GtkTreeViewColumn
    @brief Creates a new GtkTreeViewColumn with a number of default values.
    @param title The title to set the header to.
    @param cell The GtkCellRenderer.
    @param attributes An array of pairs [ attribute, column, ... ]
    @return a new GtkTreeViewColumn

    This is equivalent to calling gtk_tree_view_column_set_title(),
    gtk_tree_view_column_pack_start(), and gtk_tree_view_column_set_attributes()
    on the newly created GtkTreeViewColumn.
 */
FALCON_FUNC TreeViewColumn::new_with_attributes( VMARG )
{
    Item* i_title = vm->param( 0 );
    Item* i_cell = vm->param( 1 );
    Item* i_attr = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_title || !i_title->isString()
        || !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_attr || !i_attr->isArray() )
        throw_inv_params( "S,GtkCellRenderer,A" );
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
    GtkTreeViewColumn* view = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title( view, title.c_str() );
    Item it;
    for ( int i = 0; i < len; i += 2 )
    {
        it = attr->at( i );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isString() )
        {
            g_object_unref( view );
            throw_inv_params( "S" );
        }
#endif
        AutoCString key( it.asString() );
        it = attr->at( i + 1 );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isInteger() )
        {
            g_object_unref( view );
            throw_inv_params( "I" );
        }
#endif
        gtk_tree_view_column_add_attribute( view,
                                            cell, key.c_str(), it.asInteger() );
    }
    vm->retval( new Gtk::TreeViewColumn( vm->findWKI( "GtkTreeViewColumn" )->asClass(),
                                         view ) );
}


/*#
    @method pack_start GtkTreeViewColumn
    @brief Packs the cell into the beginning of the column.
    @param cell The GtkCellRenderer.
    @param expand TRUE if cell is to be given extra space allocated to tree_column.

    If expand is FALSE, then the cell is allocated no more space than it needs.
    Any unused space is divided evenly between cells for which expand is TRUE.
 */
FALCON_FUNC TreeViewColumn::pack_start( VMARG )
{
    Item* i_cell = vm->param( 0 );
    Item* i_exp = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_exp || !i_exp->isBoolean() )
        throw_inv_params( "GtkCellRenderer,B" );
#endif
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    gtk_tree_view_column_pack_start( GET_TREEVIEWCOLUMN( vm->self() ),
                                     cell, (gboolean) i_exp->asBoolean() );
}


/*#
    @method pack_end GtkTreeViewColumn
    @brief Adds the cell to end of the column.
    @param cell The GtkCellRenderer.
    @param expand TRUE if cell is to be given extra space allocated to tree_column.

    If expand is FALSE, then the cell is allocated no more space than it needs.
    Any unused space is divided evenly between cells for which expand is TRUE.
 */
FALCON_FUNC TreeViewColumn::pack_end( VMARG )
{
    Item* i_cell = vm->param( 0 );
    Item* i_exp = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_exp || !i_exp->isBoolean() )
        throw_inv_params( "GtkCellRenderer,B" );
#endif
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    gtk_tree_view_column_pack_end( GET_TREEVIEWCOLUMN( vm->self() ),
                                   cell, (gboolean) i_exp->asBoolean() );
}


/*#
    @method clear GtkTreeViewColumn
    @brief Unsets all the mappings on all renderers on the tree_column.
 */
FALCON_FUNC TreeViewColumn::clear( VMARG )
{
    NO_ARGS
    gtk_tree_view_column_clear( GET_TREEVIEWCOLUMN( vm->self() ) );
}


/*#
    @method add_attribute GtkTreeViewColumn
    @brief Adds an attribute mapping to the list in tree_column.
    @param cell_renderer the GtkCellRenderer to set attributes on
    @param attribute An attribute on the renderer
    @param column The column position on the model to get the attribute from.

    The column is the column of the model to get a value from, and the attribute
    is the parameter on cell_renderer to be set from the value. So for example
    if column 2 of the model contains strings, you could have the "text"
    attribute of a GtkCellRendererText get its values from column 2.
 */
FALCON_FUNC TreeViewColumn::add_attribute( VMARG )
{
    Item* i_cell = vm->param( 0 );
    Item* i_attr = vm->param( 1 );
    Item* i_col = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_attr || !i_attr->isString()
        || !i_col || !i_col->isInteger() )
        throw_inv_params( "GtkCellRenderer,S,I" );
#endif
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    AutoCString attr( i_attr->asString() );
    gtk_tree_view_column_add_attribute( GET_TREEVIEWCOLUMN( vm->self() ),
                                        cell, attr.c_str(), i_col->asInteger() );
}


/*#
    @method set_attributes GtkTreeViewColumn
    @brief Sets the attributes in the list as the attributes of tree_column.
    @param cell_renderer the GtkCellRenderer we're setting the attributes of
    @param attributes An array of pairs [ attribute, column, ... ]

    The attributes should be in attribute/column order, as in
    gtk_tree_view_column_add_attribute(). All existing attributes are removed,
    and replaced with the new attributes.
 */
FALCON_FUNC TreeViewColumn::set_attributes( VMARG )
{
    Item* i_cell = vm->param( 0 );
    Item* i_attr = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_attr || !i_attr->isArray() )
        throw_inv_params( "GtkCellRenderer,A" );
#endif
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    CoreArray* attr = i_attr->asArray();
    const int len = attr->length();
#ifndef NO_PARAMETER_CHECK
    if ( len == 0 )
        throw_inv_params( "Non-empty array" );
    if ( len % 2 != 0 )
        throw_inv_params( "Array of pairs" );
#endif
    MYSELF;
    GET_OBJ( self );
    // if an error occurs after, view is already cleared!
    gtk_tree_view_column_clear_attributes( (GtkTreeViewColumn*)_obj, cell );
    Item it;
    for ( int i = 0; i < len; i += 2 )
    {
        it = attr->at( i );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isString() )
            throw_inv_params( "S" );
#endif
        AutoCString key( it.asString() );
        it = attr->at( i + 1 );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isInteger() )
            throw_inv_params( "I" );
#endif
        gtk_tree_view_column_add_attribute( (GtkTreeViewColumn*)_obj,
                                            cell, key.c_str(), it.asInteger() );
    }
}


/*#
    @method set_cell_data_func GtkTreeViewColumn
    @brief Sets the GtkTreeViewColumnFunc to use for the column.
    @param cell_renderer A GtkCellRenderer
    @param func The GtkTreeViewColumnFunc to use, or nil.
    @param func_data The user data for func, or nil.

    This function is used instead of the standard attributes mapping for setting
    the column value, and should set the value of tree_column's cell renderer as
    appropriate. func may be NULL to remove an older one.
 */
FALCON_FUNC TreeViewColumn::set_cell_data_func( VMARG )
{
    Item* i_cell = vm->param( 0 );
    Item* i_func = vm->param( 1 );
    Item* i_data = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer )
        || !i_func || !( i_func->isNil() || i_func->isCallable() )
        || !i_data )
        throw_inv_params( "GtkCellRenderer,[C],[X]" );
#endif
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    MYSELF;
    GET_OBJ( self );
    if ( i_func->isNil() )
    {
        if ( g_object_get_data( (GObject*)_obj,
                                "__tree_view_column_cell_data_func__" ) )
        {
            g_object_set_data( (GObject*)_obj,
                               "__tree_view_column_cell_data_func__", NULL );
            g_object_set_data( (GObject*)_obj,
                               "__tree_view_column_cell_data_func_data__", NULL );
        }
        gtk_tree_view_column_set_cell_data_func( (GtkTreeViewColumn*)_obj,
                                                 cell, NULL, NULL, NULL );
    }
    else
    {
        g_object_set_data_full( (GObject*)_obj,
                                "__tree_view_column_cell_data_func__",
                                new GarbageLock( *i_func ),
                                &CoreGObject::release_lock );
        g_object_set_data_full( (GObject*)_obj,
                                "__tree_view_column_cell_data_func_data__",
                                new GarbageLock( *i_data ),
                                &CoreGObject::release_lock );
        gtk_tree_view_column_set_cell_data_func( (GtkTreeViewColumn*)_obj,
                                                 cell,
                                                 &TreeViewColumn::exec_cell_data_func,
                                                 (gpointer) vm,
                                                 NULL );
    }
}


void TreeViewColumn::exec_cell_data_func( GtkTreeViewColumn* tree_column,
                                          GtkCellRenderer* cell,
                                          GtkTreeModel* tree_model,
                                          GtkTreeIter* iter,
                                          gpointer _vm )
{
    GarbageLock* func_lock = (GarbageLock*) g_object_get_data( (GObject*) tree_column,
                                        "__tree_view_column_cell_data_func__" );
    GarbageLock* data_lock = (GarbageLock*) g_object_get_data( (GObject*) tree_column,
                                        "__tree_view_column_cell_data_func_data__" );
    assert( func_lock && data_lock );
    Item func = func_lock->item();
    Item data = func_lock->item();
    VMachine* vm = (VMachine*) _vm;
    vm->pushParam( new Gtk::CellRenderer( vm->findWKI( "GtkCellRenderer" )->asClass(), cell ) );
    vm->pushParam( new Gtk::TreeModel( vm->findWKI( "GtkTreeModel" )->asClass(), tree_model ) );
    vm->pushParam( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), iter ) );
    vm->pushParam( data );
    vm->callItem( func, 4 );
}


/*#
    @method clear_attributes GtkTreeViewColumn
    @brief Clears all existing attributes previously set with gtk_tree_view_column_set_attributes().
    @param cell_renderer a GtkCellRenderer to clear the attribute mapping on.
 */
FALCON_FUNC TreeViewColumn::clear_attributes( VMARG )
{
    Item* i_cell = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer ) )
        throw_inv_params( "GtkCellRenderer" );
#endif
    GtkCellRenderer* cell = GET_CELLRENDERER( *i_cell );
    gtk_tree_view_column_clear_attributes( GET_TREEVIEWCOLUMN( vm->self() ), cell );
}


/*#
    @method set_spacing GtkTreeViewColumn
    @brief Sets the spacing field of tree_column, which is the number of pixels to place between cell renderers packed into it.
    @param distance between cell renderers in pixels.
 */
FALCON_FUNC TreeViewColumn::set_spacing( VMARG )
{
    Item* i_dist = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_dist || !i_dist->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_tree_view_column_set_spacing( GET_TREEVIEWCOLUMN( vm->self() ), i_dist->asInteger() );
}


/*#
    @method get_spacing GtkTreeViewColumn
    @brief Returns the spacing of tree_column.
    @return the spacing of tree_column.
 */
FALCON_FUNC TreeViewColumn::get_spacing( VMARG )
{
    NO_ARGS
    vm->retval( gtk_tree_view_column_get_spacing( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_visible GtkTreeViewColumn
    @brief Sets the visibility of tree_column.
    @param visible TRUE if the tree_column is visible.
 */
FALCON_FUNC TreeViewColumn::set_visible( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_column_set_visible( GET_TREEVIEWCOLUMN( vm->self() ),
                                      (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_visible GtkTreeViewColumn
    @brief Returns TRUE if tree_column is visible.
    @return whether the column is visible or not. If it is visible, then the tree will show the column.
 */
FALCON_FUNC TreeViewColumn::get_visible( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_column_get_visible( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_resizable GtkTreeViewColumn
    @brief If resizable is TRUE, then the user can explicitly resize the column by grabbing the outer edge of the column button.
    @param resizeable TRUE, if the column can be resized

    If resizable is TRUE and sizing mode of the column is GTK_TREE_VIEW_COLUMN_AUTOSIZE,
    then the sizing mode is changed to GTK_TREE_VIEW_COLUMN_GROW_ONLY.
 */
FALCON_FUNC TreeViewColumn::set_resizable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_column_set_resizable( GET_TREEVIEWCOLUMN( vm->self() ),
                                        (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_resizable GtkTreeViewColumn
    @brief Returns TRUE if the tree_column can be resized by the end user.
    @return TRUE, if the tree_column can be resized.
 */
FALCON_FUNC TreeViewColumn::get_resizable( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_column_get_resizable( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_sizing GtkTreeViewColumn
    @brief Sets the growth behavior of tree_column to type.
    @param type The GtkTreeViewColumnSizing.
 */
FALCON_FUNC TreeViewColumn::set_sizing( VMARG )
{
    Item* i_siz = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_siz || !i_siz->isInteger() )
        throw_inv_params( "GtkTreeViewColumnSizing" );
#endif
    gtk_tree_view_column_set_sizing( GET_TREEVIEWCOLUMN( vm->self() ),
                                     (GtkTreeViewColumnSizing) i_siz->asInteger() );
}


/*#
    @method get_sizing GtkTreeViewColumn
    @brief Returns the current type of tree_column.
    @return The type of tree_column (GtkTreeViewColumnSizing).
 */
FALCON_FUNC TreeViewColumn::get_sizing( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_tree_view_column_get_sizing( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method get_width GtkTreeViewColumn
    @brief Returns the current size of tree_column in pixels.
    @return The current width of tree_column.
 */
FALCON_FUNC TreeViewColumn::get_width( VMARG )
{
    NO_ARGS
    vm->retval( gtk_tree_view_column_get_width( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method get_fixed_width GtkTreeViewColumn
    @brief Gets the fixed width of the column.

    This value is only meaning may not be the actual width of the column on
    the screen, just what is requested.
 */
FALCON_FUNC TreeViewColumn::get_fixed_width( VMARG )
{
    NO_ARGS
    vm->retval( gtk_tree_view_column_get_fixed_width( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_fixed_width GtkTreeViewColumn
    @brief Sets the size of the column in pixels.
    @param fixed_width The size to set tree_column to. Must be greater than 0.

    This is meaningful only if the sizing type is GTK_TREE_VIEW_COLUMN_FIXED
    The size of the column is clamped to the min/max width for the column.
    Please note that the min/max width of the column doesn't actually affect
    the "fixed_width" property of the widget, just the actual size when displayed.
 */
FALCON_FUNC TreeViewColumn::set_fixed_width( VMARG )
{
    Item* i_w = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || !i_w->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_tree_view_column_set_fixed_width( GET_TREEVIEWCOLUMN( vm->self() ),
                                          i_w->asInteger() );
}


/*#
    @method set_min_width GtkTreeViewColumn
    @brief Sets the minimum width of the tree_column.
    @param min_width The minimum width of the column in pixels, or -1.

    If min_width is -1, then the minimum width is unset.
 */
FALCON_FUNC TreeViewColumn::set_min_width( VMARG )
{
    Item* i_w = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || !i_w->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_tree_view_column_set_min_width( GET_TREEVIEWCOLUMN( vm->self() ),
                                        i_w->asInteger() );
}


/*#
    @method get_min_width GtkTreeViewColumn
    @brief Returns the minimum width in pixels of the tree_column, or -1 if no minimum width is set.
    @return The minimum width of the tree_column.
 */
FALCON_FUNC TreeViewColumn::get_min_width( VMARG )
{
    NO_ARGS
    vm->retval( gtk_tree_view_column_get_min_width( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_max_width GtkTreeViewColumn
    @brief Sets the maximum width of the tree_column.
    @param max_width The maximum width of the column in pixels, or -1.

    If max_width is -1, then the maximum width is unset. Note, the column can
    actually be wider than max width if it's the last column in a view. In
    this case, the column expands to fill any extra space.
 */
FALCON_FUNC TreeViewColumn::set_max_width( VMARG )
{
    Item* i_w = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || !i_w->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_tree_view_column_set_max_width( GET_TREEVIEWCOLUMN( vm->self() ),
                                        i_w->asInteger() );
}


/*#
    @method get_max_width GtkTreeViewColumn
    @brief Returns the maximum width in pixels of the tree_column, or -1 if no maximum width is set.
    @return The maximum width of the tree_column.
 */
FALCON_FUNC TreeViewColumn::get_max_width( VMARG )
{
    NO_ARGS
    vm->retval( gtk_tree_view_column_get_max_width( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method clicked GtkTreeViewColumn
    @brief Emits the "clicked" signal on the column.

    This function will only work if tree_column is clickable.
 */
FALCON_FUNC TreeViewColumn::clicked( VMARG )
{
    NO_ARGS
    gtk_tree_view_column_clicked( GET_TREEVIEWCOLUMN( vm->self() ) );
}


/*#
    @method set_title GtkTreeViewColumn
    @brief Sets the title of the tree_column.
    @param title The title of the tree_column.

    If a custom widget has been set, then this value is ignored.
 */
FALCON_FUNC TreeViewColumn::set_title( VMARG )
{
    Item* i_title = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_title || !i_title->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString title( i_title->asString() );
    gtk_tree_view_column_set_title( GET_TREEVIEWCOLUMN( vm->self() ),
                                    title.c_str() );
}


/*#
    @method get_title GtkTreeViewColumn
    @brief Returns the title of the widget.
    @return the title of the column.
 */
FALCON_FUNC TreeViewColumn::get_title( VMARG )
{
    NO_ARGS
    const gchar* title = gtk_tree_view_column_get_title( GET_TREEVIEWCOLUMN( vm->self() ) );
    if ( title )
        vm->retval( UTF8String( title ) );
    else
        vm->retnil();
}


/*#
    @method set_expand GtkTreeViewColumn
    @brief Sets the column to take available extra space.
    @param expand TRUE if the column should take available extra space, FALSE if not

    This space is shared equally amongst all columns that have the expand set to
    TRUE. If no column has this option set, then the last column gets all extra
    space. By default, every column is created with this FALSE.
 */
FALCON_FUNC TreeViewColumn::set_expand( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_column_set_expand( GET_TREEVIEWCOLUMN( vm->self() ),
                                     (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_expand GtkTreeViewColumn
    @brief Return TRUE if the column expands to take any available space.
    @return TRUE, if the column expands
 */
FALCON_FUNC TreeViewColumn::get_expand( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_column_get_expand( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_clickable GtkTreeViewColumn
    @brief Sets the header to be active if active is TRUE.
    @param clickable TRUE if the header is active.

    When the header is active, then it can take keyboard focus, and can be clicked.
 */
FALCON_FUNC TreeViewColumn::set_clickable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_column_set_clickable( GET_TREEVIEWCOLUMN( vm->self() ),
                                        (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_clickable GtkTreeViewColumn
    @brief Returns TRUE if the user can click on the header for the column.
    @return TRUE if user can click the column header.
 */
FALCON_FUNC TreeViewColumn::get_clickable( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_column_get_clickable( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_widget GtkTreeViewColumn
    @brief Sets the widget in the header to be widget.
    @param widget A child GtkWidget, or NULL.

    If widget is NULL, then the header button is set with a GtkLabel set to the
    title of tree_column.
 */
FALCON_FUNC TreeViewColumn::set_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !( i_wdt->isNil() || ( i_wdt->isObject()
        && IS_DERIVED( i_wdt, GtkWidget ) ) ) )
        throw_inv_params( "[GtkWidget]" );
#endif
    gtk_tree_view_column_set_widget( GET_TREEVIEWCOLUMN( vm->self() ),
                                     i_wdt->isNil() ? NULL : GET_WIDGET( *i_wdt ) );
}


/*#
    @method get_widget GtkTreeViewColumn
    @brief Returns the GtkWidget in the button on the column header.
    @return The GtkWidget in the column header, or NULL

    If a custom widget has not been set then NULL is returned.
 */
FALCON_FUNC TreeViewColumn::get_widget( VMARG )
{
    NO_ARGS
    GtkWidget* wdt = gtk_tree_view_column_get_widget( GET_TREEVIEWCOLUMN( vm->self() ) );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method set_alignment GtkTreeViewColumn
    @brief Sets the alignment of the title or custom widget inside the column header.
    @param xalign The alignment, which is between [0.0 and 1.0] inclusive.

    The alignment determines its location inside the button -- 0.0 for left,
    0.5 for center, 1.0 for right.
 */
FALCON_FUNC TreeViewColumn::set_alignment( VMARG )
{
    Item* i_xalign = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_xalign || !i_xalign->isOrdinal() )
        throw_inv_params( "N" );
#endif
    gtk_tree_view_column_set_alignment( GET_TREEVIEWCOLUMN( vm->self() ),
                                        i_xalign->forceNumeric() );
}


/*#
    @method get_alignment GtkTreeViewColumn
    @brief Returns the current x alignment of tree_column.

    This value can range between 0.0 and 1.0.
 */
FALCON_FUNC TreeViewColumn::get_alignment( VMARG )
{
    NO_ARGS
    vm->retval( (numeric) gtk_tree_view_column_get_alignment( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_reorderable GtkTreeViewColumn
    @brief If reorderable is TRUE, then the column can be reordered by the end user dragging the header.
    @param reorderable TRUE, if the column can be reordered.
 */
FALCON_FUNC TreeViewColumn::set_reorderable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_column_set_reorderable( GET_TREEVIEWCOLUMN( vm->self() ),
                                          (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_reorderable GtkTreeViewColumn
    @brief Returns TRUE if the tree_column can be reordered by the user.
    @return TRUE if the tree_column can be reordered by the user.
 */
FALCON_FUNC TreeViewColumn::get_reorderable( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_column_get_reorderable( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_sort_column_id GtkTreeViewColumn
    @brief Sets the logical sort_column_id that this column sorts on when this column is selected for sorting.
    @param sort_column_id The sort_column_id of the model to sort on.

    Doing so makes the column header clickable.
 */
FALCON_FUNC TreeViewColumn::set_sort_column_id( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || !i_id->isInteger() )
        throw_inv_params( "I" );
#endif
    gtk_tree_view_column_set_sort_column_id( GET_TREEVIEWCOLUMN( vm->self() ),
                                             i_id->asInteger() );
}


/*#
    @method get_sort_column_id GtkTreeViewColumn
    @brief Gets the logical sort_column_id that the model sorts on when this column is selected for sorting.
    @return the current sort_column_id for this column, or -1 if this column can't be used for sorting.

    See gtk_tree_view_column_set_sort_column_id().
 */
FALCON_FUNC TreeViewColumn::get_sort_column_id( VMARG )
{
    NO_ARGS
    vm->retval( gtk_tree_view_column_get_sort_column_id( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_sort_indicator GtkTreeViewColumn
    @brief Call this function with a setting of TRUE to display an arrow in the header button indicating the column is sorted.
    @param setting TRUE to display an indicator that the column is sorted

    Call gtk_tree_view_column_set_sort_order() to change the direction of the arrow.
 */
FALCON_FUNC TreeViewColumn::set_sort_indicator( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_tree_view_column_set_sort_indicator( GET_TREEVIEWCOLUMN( vm->self() ),
                                             (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_sort_indicator GtkTreeViewColumn
    @brief Gets the value set by gtk_tree_view_column_set_sort_indicator().
    @return whether the sort indicator arrow is displayed
 */
FALCON_FUNC TreeViewColumn::get_sort_indicator( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_column_get_sort_indicator( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method set_sort_order GtkTreeViewColumn
    @brief Changes the appearance of the sort indicator.
    @param order sort order that the sort indicator should indicate (GtkSortType)

    This does not actually sort the model. Use gtk_tree_view_column_set_sort_column_id()
    if you want automatic sorting support. This function is primarily for custom
    sorting behavior, and should be used in conjunction with
    gtk_tree_sortable_set_sort_column() to do that. For custom models, the
    mechanism will vary.

    The sort indicator changes direction to indicate normal sort or reverse sort.
    Note that you must have the sort indicator enabled to see anything when
    calling this function; see gtk_tree_view_column_set_sort_indicator().
 */
FALCON_FUNC TreeViewColumn::set_sort_order( VMARG )
{
    Item* i_ord = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ord || !i_ord->isInteger() )
        throw_inv_params( "GtkSortType" );
#endif
    gtk_tree_view_column_set_sort_order( GET_TREEVIEWCOLUMN( vm->self() ),
                                         (GtkSortType) i_ord->asInteger() );
}


/*#
    @method get_sort_order GtkTreeViewColumn
    @brief Gets the value set by gtk_tree_view_column_set_sort_order().
    @return the sort order the sort indicator is indicating (GtkSortType)
 */
FALCON_FUNC TreeViewColumn::get_sort_order( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gtk_tree_view_column_get_sort_order( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method cell_set_cell_data GtkTreeViewColumn
    @brief Sets the cell renderer based on the tree_model and iter.
    @param tree_model The GtkTreeModel to to get the cell renderers attributes from.
    @param iter The GtkTreeIter to to get the cell renderer's attributes from.
    @param is_expander TRUE, if the row has children
    @param is_expanded TRUE, if the row has visible children

    That is, forevery attribute mapping in tree_column, it will get a value from
    the set column on the iter, and use that value to set the attribute on the
    cell renderer. This is used primarily by the GtkTreeView.
 */
FALCON_FUNC TreeViewColumn::cell_set_cell_data( VMARG )
{
    Item* i_mdl = vm->param( 0 );
    Item* i_iter = vm->param( 1 );
    Item* i_expander = vm->param( 2 );
    Item* i_expanded = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mdl || !i_mdl->isObject() || !IS_DERIVED( i_mdl, GtkTreeModel )
        || !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_expander || !i_expander->isBoolean()
        || !i_expanded || !i_expanded->isBoolean() )
        throw_inv_params( "GtkTreeModel,GtkTreeIter,B,B" );
#endif
    GtkTreeModel* mdl = GET_TREEMODEL( *i_mdl );
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    gtk_tree_view_column_cell_set_cell_data( GET_TREEVIEWCOLUMN( vm->self() ),
                                             mdl,
                                             iter,
                                             (gboolean) i_expander->asBoolean(),
                                             (gboolean) i_expanded->asBoolean() );
}


/*#
    @method cell_get_size GtkTreeViewColumn
    @brief Obtains the width and height needed to render the column.
    @param cell_area The area (GdkRectangle) a cell in the column will be allocated, or NULL.
    @return an array [ x offset, y offset of a cell relative to cell_area, width, height needed to render a cell ]

    This is used primarily by the GtkTreeView.
 */
FALCON_FUNC TreeViewColumn::cell_get_size( VMARG )
{
    Item* i_cell = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !( i_cell->isNil() || ( i_cell->isObject()
        && IS_DERIVED( i_cell, GdkRectangle ) ) ) )
        throw_inv_params( "[GdkRectangle]" );
#endif
    gint x, y, w, h;
    gtk_tree_view_column_cell_get_size( GET_TREEVIEWCOLUMN( vm->self() ),
                                        i_cell->isNil() ? NULL : GET_RECTANGLE( *i_cell ),
                                        &x, &y, &w, &h );
    CoreArray* arr = new CoreArray( 4 );
    arr->append( x );
    arr->append( y );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method cell_get_position GtkTreeViewColumn
    @brief Obtains the horizontal position and size of a cell in a column.
    @param cell a GtkCellRenderer
    @return an array [ horizontal position of cell within tree_column, width of cell ], or nil

    If the cell is not found in the column, nil is returned.
 */
FALCON_FUNC TreeViewColumn::cell_get_position( VMARG )
{
    Item* i_cell = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer ) )
        throw_inv_params( "GtkCellRenderer" );
#endif
    gint start_pos, width;
    if ( gtk_tree_view_column_cell_get_position( GET_TREEVIEWCOLUMN( vm->self() ),
                                                 GET_CELLRENDERER( *i_cell ),
                                                 &start_pos,
                                                 &width ) )
    {
        CoreArray* arr = new CoreArray( 2 );
        arr->append( start_pos );
        arr->append( width );
        vm->retval( arr );
    }
    else
        vm->retnil();
}


/*#
    @method cell_is_visible GtkTreeViewColumn
    @brief Returns TRUE if any of the cells packed into the tree_column are visible.
    @return TRUE, if any of the cells packed into the tree_column are currently visible

    For this to be meaningful, you must first initialize the cells with
    gtk_tree_view_column_cell_set_cell_data()
 */
FALCON_FUNC TreeViewColumn::cell_is_visible( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_tree_view_column_cell_is_visible( GET_TREEVIEWCOLUMN( vm->self() ) ) );
}


/*#
    @method focus_cell GtkTreeViewColumn
    @brief Sets the current keyboard focus to be at cell, if the column contains 2 or more editable and activatable cells.
    @param cell A GtkCellRenderer
 */
FALCON_FUNC TreeViewColumn::focus_cell( VMARG )
{
    Item* i_cell = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_cell || !i_cell->isObject() || !IS_DERIVED( i_cell, GtkCellRenderer ) )
        throw_inv_params( "GtkCellRenderer" );
#endif
    gtk_tree_view_column_focus_cell( GET_TREEVIEWCOLUMN( vm->self() ),
                                     GET_CELLRENDERER( *i_cell ) );
}


/*#
    @method queue_resize GtkTreeViewColumn
    @brief Flags the column, and the cell renderers added to this column, to have their sizes renegotiated.
 */
FALCON_FUNC TreeViewColumn::queue_resize( VMARG )
{
    NO_ARGS
    gtk_tree_view_column_queue_resize( GET_TREEVIEWCOLUMN( vm->self() ) );
}


/*#
    @method get_tree_view GtkTreeViewColumn
    @brief Returns the GtkTreeView wherein tree_column has been inserted.
    @return The tree view wherein column has been inserted if any, NULL otherwise.

    If column is currently not inserted in any tree view, NULL is returned.
 */
FALCON_FUNC TreeViewColumn::get_tree_view( VMARG )
{
    NO_ARGS
    GtkWidget* view = gtk_tree_view_column_get_tree_view( GET_TREEVIEWCOLUMN( vm->self() ) );
    if ( view )
        vm->retval( new Gtk::TreeView( vm->findWKI( "GtkTreeView" )->asClass(),
                                       (GtkTreeView*) view ) );
    else
        vm->retnil();
}


} // Gtk
} // Falcon
