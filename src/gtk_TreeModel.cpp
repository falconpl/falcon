/**
 *  \file gtk_TreeModel.cpp
 */

#include "gtk_TreeModel.hpp"

#include "g_Object.hpp"
#include "gtk_TreeIter.hpp"
#include "gtk_TreePath.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeModel::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeModel = mod->addClass( "%GtkTreeModel" );

    c_TreeModel->setWKS( true );
    c_TreeModel->getClassDef()->factory( &TreeModel::factory );

    TreeModel::clsInit( mod, c_TreeModel );
}


/**
 *  \brief interface loader
 */
void TreeModel::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "signal_row_changed", &TreeModel::signal_row_changed },
    { "signal_row_deleted", &TreeModel::signal_row_deleted },
    { "signal_row_has_child_toggled",&TreeModel::signal_row_has_child_toggled },
    { "signal_row_inserted",&TreeModel::signal_row_inserted },
    { "signal_rows_reordered",&TreeModel::signal_rows_reordered },
    { "get_flags",          &TreeModel::get_flags },
    { "get_n_columns",      &TreeModel::get_n_columns },
    { "get_column_type",    &TreeModel::get_column_type },
    { "get_iter",           &TreeModel::get_iter },
    { "get_iter_from_string",&TreeModel::get_iter_from_string },
    { "get_iter_first",     &TreeModel::get_iter_first },
    { "get_path",           &TreeModel::get_path },
    { "get_value",          &TreeModel::get_value },
    { "iter_next",          &TreeModel::iter_next },
    { "iter_children",      &TreeModel::iter_children },
    { "iter_has_child",     &TreeModel::iter_has_child },
    { "iter_n_children",    &TreeModel::iter_n_children },
    { "iter_nth_child",     &TreeModel::iter_nth_child },
    { "iter_parent",        &TreeModel::iter_parent },
    { "get_string_from_iter",&TreeModel::get_string_from_iter },
#if 0
    { "ref_node",           &TreeModel::ref_node },
    { "unref_node",         &TreeModel::unref_node },
    { "get",                &TreeModel::get },
    { "get_valist",         &TreeModel::get_valist },
    { "foreach",            &TreeModel::foreach_ },
#endif
    { "row_changed",        &TreeModel::row_changed },
    { "row_inserted",       &TreeModel::row_inserted },
    { "row_has_child_toggled",&TreeModel::row_has_child_toggled },
    { "row_deleted",        &TreeModel::row_has_child_toggled },
    { "rows_reordered",     &TreeModel::rows_reordered },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


TreeModel::TreeModel( const Falcon::CoreClass* gen, const GtkTreeModel* model )
    :
    Gtk::CoreGObject( gen, (GObject*) model )
{}


Falcon::CoreObject* TreeModel::factory( const Falcon::CoreClass* gen, void* model, bool )
{
    return new TreeModel( gen, (GtkTreeModel*) model );
}


bool TreeModel::implementedBy( const Falcon::Item* it )
{
    if (   IS_DERIVED( it, GtkListStore )
        || IS_DERIVED( it, GtkTreeModelFilter )
        || IS_DERIVED( it, GtkTreeModelSort )
        || IS_DERIVED( it, GtkTreeStore ) )
        return true;
    return false;
}


/*#
    @class GtkTreeModel
    @brief The GtkTreeModel interface defines a generic tree interface for use by the GtkTreeView widget.

    [...]
 */


/*#
    @method signal_row_changed GtkTreeModel
    @brief This signal is emitted when a row in the model has changed.
 */
FALCON_FUNC TreeModel::signal_row_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "row_changed", (void*) &TreeModel::on_row_changed, vm );
}


void TreeModel::on_row_changed( GtkTreeModel* obj, GtkTreePath* path,
                                GtkTreeIter* titer, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "row_changed", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkPath = vm->findWKI( "GtkTreePath" );
    Item* wkIter = vm->findWKI( "GtkTreeIter" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_row_changed", it ) )
            {
                printf(
                "[GtkTreeModel::on_row_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreePath( wkPath->asClass(), path ) );
        vm->pushParam( new Gtk::TreeIter( wkIter->asClass(), titer ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
    // free params?..
    //gtk_tree_path_free( path );
    //gtk_tree_iter_free( iter );
}


/*#
    @method signal_row_deleted GtkTreeModel
    @brief This signal is emitted when a row has been deleted.

    Note that no iterator is passed to the signal handler, since the row is
    already deleted.

    Implementations of GtkTreeModel must emit row-deleted before removing the node
    from its internal data structures. This is because models and views which
    access and monitor this model might have references on the node which need
    to be released in the row-deleted handler.
 */
FALCON_FUNC TreeModel::signal_row_deleted( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "row_deleted", (void*) &TreeModel::on_row_deleted, vm );
}


void TreeModel::on_row_deleted( GtkTreeModel* obj, GtkTreePath* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "row_deleted", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkPath = vm->findWKI( "GtkTreePath" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_row_deleted", it ) )
            {
                printf(
                "[GtkTreeModel::on_row_deleted] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreePath( wkPath->asClass(), path ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
    // free params?..
    //gtk_tree_path_free( path );
}


/*#
    @method row_has_child_toggled GtkTreeModel
    @brief This signal is emitted when a row has gotten the first child row or lost its last child row.
 */
FALCON_FUNC TreeModel::signal_row_has_child_toggled( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "row_has_child_toggled",
                             (void*) &TreeModel::on_row_has_child_toggled, vm );
}


void TreeModel::on_row_has_child_toggled( GtkTreeModel* obj, GtkTreePath* path,
                                          GtkTreeIter* titer, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "row_has_child_toggled", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkPath = vm->findWKI( "GtkTreePath" );
    Item* wkIter = vm->findWKI( "GtkTreeIter" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_row_has_child_toggled", it ) )
            {
                printf(
                "[GtkTreeModel::on_row_has_child_toggled] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreePath( wkPath->asClass(), path ) );
        vm->pushParam( new Gtk::TreeIter( wkIter->asClass(), titer ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
    // free params?..
    //gtk_tree_path_free( path );
    //gtk_tree_iter_free( iter );
}


/*#
    @method signal_row_inserted GtkTreeModel
    @brief This signal is emitted when a new row has been inserted in the model.

    Note that the row may still be empty at this point, since it is a common
    pattern to first insert an empty row, and then fill it with the desired values.
 */
FALCON_FUNC TreeModel::signal_row_inserted( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "row_inserted", (void*) &TreeModel::on_row_inserted, vm );
}


void TreeModel::on_row_inserted( GtkTreeModel* obj, GtkTreePath* path,
                                 GtkTreeIter* titer, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "row_inserted", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wkPath = vm->findWKI( "GtkTreePath" );
    Item* wkIter = vm->findWKI( "GtkTreeIter" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_row_inserted", it ) )
            {
                printf(
                "[GtkTreeModel::on_row_inserted] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreePath( wkPath->asClass(), path ) );
        vm->pushParam( new Gtk::TreeIter( wkIter->asClass(), titer ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
    // free params?..
    //gtk_tree_path_free( path );
    //gtk_tree_iter_free( iter );
}


/*#
    @method signal_rows_reordered GtkTreeModel
    @brief This signal is emitted when the children of a node in the GtkTreeModel have been reordered.

    Note that this signal is not emitted when rows are reordered by DND,
    since this is implemented by removing and then reinserting the row.
 */
FALCON_FUNC TreeModel::signal_rows_reordered( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "rows_reordered", (void*) &TreeModel::on_rows_reordered, vm );
}


void TreeModel::on_rows_reordered( GtkTreeModel* obj, GtkTreePath* path,
                                   GtkTreeIter* titer, gpointer order, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "rows_reordered", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    gint* norder = (gint*) order;
    Iterator iter( cs );
    Item it;
    Item* wkPath = vm->findWKI( "GtkTreePath" );
    Item* wkIter = vm->findWKI( "GtkTreeIter" );

    int i, cnt = 0;
    for ( i = 0; norder[i] != -1; ++i ) ++cnt;
    CoreArray arr( cnt );
    for ( i = 0; i < cnt; ++i )
        arr.append( norder[i] );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_rows_reordered", it ) )
            {
                printf(
                "[GtkTreeModel::on_rows_reordered] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TreePath( wkPath->asClass(), path ) );
        vm->pushParam( new Gtk::TreeIter( wkIter->asClass(), titer ) );
        vm->pushParam( new CoreArray( arr ) );
        vm->callItem( it, 3 );
    }
    while ( iter.hasCurrent() );
    // free params?..
    //gtk_tree_path_free( path );
    //gtk_tree_iter_free( iter );
}


/*#
    @method get_flags GtkTreeModel
    @brief Returns a set of flags supported by this interface.
    @return The flags supported by this interface.

    The flags are a bitwise combination of GtkTreeModelFlags. The flags
    supported should not change during the lifecycle of the tree_model.
 */
FALCON_FUNC TreeModel::get_flags( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tree_model_get_flags( (GtkTreeModel*)_obj ) );
}


/*#
    @method get_n_columns GtkTreeModel
    @brief Returns the number of columns supported by tree_model.
    @return The number of columns.
 */
FALCON_FUNC TreeModel::get_n_columns( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_tree_model_get_n_columns( (GtkTreeModel*)_obj ) );
}


/*#
    @method get_column_type GtkTreeModel
    @brief Returns the type of the column.
    @param index The column index.
    @return The type of the column (GType).
 */
FALCON_FUNC TreeModel::get_column_type( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_tree_model_get_column_type( (GtkTreeModel*)_obj,
                                                        i_idx->asInteger() ) );
}


/*#
    @method get_iter GtkTreeModel
    @brief Returns a valid iterator pointing to path.
    @param path The GtkTreePath.
    @return a GtkTreeIter
    @raise ParamError if path is invalid
 */
FALCON_FUNC TreeModel::get_iter( VMARG )
{
    Item* i_path = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    MYSELF;
    GET_OBJ( self );
    GtkTreeIter iter;
    gboolean ret = gtk_tree_model_get_iter( (GtkTreeModel*)_obj, &iter, path );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(),
                                       &iter ) );
    else
        throw_inv_params( "GtkTreePath" );
}


/*#
    @method get_iter_from_string GtkTreeModel
    @brief Returns a valid iterator pointing to path_string, if it exists.
    @param path_string A string representation of a GtkTreePath.
    @return a GtkTreeIter.
    @raise ParamError if path is invalid
 */
FALCON_FUNC TreeModel::get_iter_from_string( VMARG )
{
    Item* i_pth = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pth || !i_pth->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString pth( i_pth->asString() );
    MYSELF;
    GET_OBJ( self );
    GtkTreeIter iter;
    gboolean ret = gtk_tree_model_get_iter_from_string( (GtkTreeModel*)_obj, &iter,
                                                        pth.c_str() );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
    else
        throw_inv_params( "S" );
}


/*#
    @method get_iter_first GtkTreeModel
    @brief Returns the first iterator in the tree (the one at the path "0") or Nil if the tree is empty.
    @return a GtkTreeIter, or nil.
 */
FALCON_FUNC TreeModel::get_iter_first( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTreeIter iter;
    gboolean ret = gtk_tree_model_get_iter_first( (GtkTreeModel*)_obj, &iter );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
    else
        vm->retnil();
}


/*#
    @method get_path GtkTreeModel
    @brief Returns a newly-created GtkTreePath referenced by iter.
    @param iter The GtkTreeIter.
    @return a newly-created GtkTreePath.
 */
FALCON_FUNC TreeModel::get_path( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    GtkTreePath* path = gtk_tree_model_get_path( (GtkTreeModel*)_obj, iter );
    vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(), path,
                                   true ) );
}


/*#
    @method get_value GtkTreeModel
    @brief Returns value to that at column.
    @param iter The GtkTreeIter.
    @param column The column to lookup the value at.
    @return the value
 */
FALCON_FUNC TreeModel::get_value( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_col = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_col || !i_col->isInteger() )
        throw_inv_params( "GtkTreeIter,I" );
#endif
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    GValue val;
    memset( &val, 0, sizeof( GValue ) );
    gtk_tree_model_get_value( (GtkTreeModel*)_obj, iter, i_col->asInteger(), &val );
    switch ( G_VALUE_TYPE( &val ) )
    { // todo: make it a function somewhere.
    case G_TYPE_NONE:
        vm->retnil();
        break;
    case G_TYPE_BOOLEAN:
        vm->retval( (bool) g_value_get_boolean( &val ) );
        break;
    case G_TYPE_INT:
        vm->retval( (int64) g_value_get_int( &val ) );
        break;
    case G_TYPE_UINT:
        vm->retval( (int64) g_value_get_uint( &val ) );
        break;
    case G_TYPE_LONG:
        vm->retval( (int64) g_value_get_long( &val ) );
        break;
    case G_TYPE_ULONG:
        vm->retval( (int64) g_value_get_ulong( &val ) );
        break;
    case G_TYPE_INT64:
        vm->retval( (int64) g_value_get_int64( &val ) );
        break;
    case G_TYPE_UINT64:
        vm->retval( (int64) g_value_get_uint64( &val ) );
        break;
    case G_TYPE_FLOAT:
        vm->retval( (numeric) g_value_get_float( &val ) );
        break;
    case G_TYPE_DOUBLE:
        vm->retval( (numeric) g_value_get_double( &val ) );
        break;
    case G_TYPE_STRING:
        vm->retval( UTF8String( g_value_get_string( &val ) ) );
        break;
    case G_TYPE_OBJECT:
        vm->retval( new Glib::Object( vm->findWKI( "GObject" )->asClass(),
                                      (GObject*) g_value_get_object( &val ) ) );
        break;
    default:
        printf( "TreeModel::get_value: not implemented (%i)", (int)G_VALUE_TYPE( &val ) );
        g_value_unset( &val );
        throw_inv_params( "Not implemented" ); // todo: throw another type of error
    }
    g_value_unset( &val );
}


/*#
    @method iter_next GtkTreeModel
    @brief Returns an iterator pointing to the node following iter at the current level, or Nil.
    @param iter The GtkTreeIter.
    @return a new GtkTreeIter or Nil if there is no next row.

    If there is no next iter, Nil is returned.
 */
FALCON_FUNC TreeModel::iter_next( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter iter = *(GET_TREEITER( *i_iter ));
    MYSELF;
    GET_OBJ( self );
    gboolean ret = gtk_tree_model_iter_next( (GtkTreeModel*)_obj, &iter );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
    else
        vm->retnil();
}


/*#
    @method iter_children GtkTreeModel
    @brief Returns an iterator pointing to the first child of parent, or Nil.
    @param parent The GtkTreeIter, or Nil.
    @return a new GtkTreeIter, or Nil if there is no children.

    If parent is NULL returns the first node, equivalent to
    gtk_tree_model_get_iter_first (iter);
 */
FALCON_FUNC TreeModel::iter_children( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !( i_iter->isNil() || ( i_iter->isObject()
        && IS_DERIVED( i_iter, GtkTreeIter ) ) ) )
        throw_inv_params( "[GtkTreeIter]" );
#endif
    GtkTreeIter* parent = i_iter->isNil() ? NULL : GET_TREEITER( *i_iter );
    GtkTreeIter iter;
    MYSELF;
    GET_OBJ( self );
    gboolean ret = gtk_tree_model_iter_children( (GtkTreeModel*)_obj, &iter, parent );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
    else
        vm->retnil();
}


/*#
    @method iter_has_child GtkTreeModel
    @brief Returns TRUE if iter has children, FALSE otherwise.
    @param iter The GtkTreeIter to test for children.
    @return TRUE if iter has children.
 */
FALCON_FUNC TreeModel::iter_has_child( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tree_model_iter_has_child( (GtkTreeModel*)_obj, iter ) );
}


/*#
    @method iter_n_children GtkTreeModel
    @brief Returns the number of children that iter has.
    @param iter The GtkTreeIter, or NULL.
    @return The number of children of iter.

    As a special case, if iter is NULL, then the number of toplevel nodes is returned.
 */
FALCON_FUNC TreeModel::iter_n_children( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !( i_iter->isNil() || ( i_iter->isObject()
        && IS_DERIVED( i_iter, GtkTreeIter ) ) ) )
        throw_inv_params( "[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = i_iter->isNil() ? NULL : GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_tree_model_iter_n_children( (GtkTreeModel*)_obj, iter ) );
}


/*#
    @method iter_nth_child GtkTreeModel
    @brief Returns an iterator being the child of parent, using the given index.
    @param parent The GtkTreeIter to get the child from, or NULL.
    @param n Then index of the desired child.
    @return a new GtkTreeIter, or Nil if the nth child is not found.

    The first index is 0. If n is too big, or parent has no children, Nil is
    returned. As a special case, if parent is NULL, then the nth root node is set.
 */
FALCON_FUNC TreeModel::iter_nth_child( VMARG )
{
    Item* i_par = vm->param( 0 );
    Item* i_n = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) )
        || !i_n || !i_n->isInteger() )
        throw_inv_params( "[GtkTreeIter],I" );
#endif
    GtkTreeIter* parent = i_par->isNil() ? NULL : GET_TREEITER( *i_par );
    MYSELF;
    GET_OBJ( self );
    GtkTreeIter iter;
    gboolean ret = gtk_tree_model_iter_nth_child( (GtkTreeModel*)_obj,
                                                     &iter,
                                                     parent,
                                                     i_n->asInteger() );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(),
                                       &iter ) );
    else
        vm->retnil();
}


/*#
    @method iter_parent GtkTreeModel
    @brief Returns an iterator being the parent of child, or Nil.
    @param child The GtkTreeIter.
    @return a new GtkTreeIter, or Nil if child has not parent.

    If child is at the toplevel, and doesn't have a parent, then Nil is returned.
 */
FALCON_FUNC TreeModel::iter_parent( VMARG )
{
    Item* i_child = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || !i_child->isObject() || !IS_DERIVED( i_child, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* child = GET_TREEITER( *i_child );
    MYSELF;
    GET_OBJ( self );
    GtkTreeIter iter;
    gboolean ret = gtk_tree_model_iter_parent( (GtkTreeModel*)_obj, &iter, child );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
    else
        vm->retnil();
}


/*#
    @method get_string_from_iter GtkTreeModel
    @brief Generates a string representation of the iter.
    @param iter A GtkTreeIter.
    @return a string.

    This string is a ':' separated list of numbers. For example, "4:10:0:3"
    would be an acceptable return value for this string.
 */
FALCON_FUNC TreeModel::get_string_from_iter( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    gchar* s = gtk_tree_model_get_string_from_iter( (GtkTreeModel*)_obj, iter );
    vm->retval( UTF8String( s ) );
    g_free( s );
}


#if 0
FALCON_FUNC TreeModel::ref_node( VMARG );
FALCON_FUNC TreeModel::unref_node( VMARG );
FALCON_FUNC TreeModel::get( VMARG );
FALCON_FUNC TreeModel::get_valist( VMARG );
FALCON_FUNC TreeModel::foreach_( VMARG );
#endif


/*#
    @method row_changed GtkTreeModel
    @brief Emits the "row-changed" signal on tree_model.
    @param path A GtkTreePath pointing to the changed row
    @param iter A valid GtkTreeIter pointing to the changed row
 */
FALCON_FUNC TreeModel::row_changed( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_iter = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreePath,GtkTreeIter" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_row_changed( (GtkTreeModel*)_obj, path, iter );
}


/*#
    @method row_inserted GtkTreeModel
    @brief Emits the "row-inserted" signal on tree_model
    @param path A GtkTreePath pointing to the inserted row
    @param iter A valid GtkTreeIter pointing to the inserted row
 */
FALCON_FUNC TreeModel::row_inserted( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_iter = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreePath,GtkTreeIter" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_row_inserted( (GtkTreeModel*)_obj, path, iter );
}


/*#
    @method row_has_child_toggled GtkTreeModel
    @brief Emits the "row-has-child-toggled" signal on tree_model.
    @param path A GtkTreePath pointing to the changed row
    @param A valid GtkTreeIter pointing to the changed row

    This should be called by models after the child state of a node changes.
 */
FALCON_FUNC TreeModel::row_has_child_toggled( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_iter = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreePath,GtkTreeIter" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_row_has_child_toggled( (GtkTreeModel*)_obj, path, iter );
}


/*#
    @method row_deleted GtkTreeModel
    @brief Emits the "row-deleted" signal on tree_model.
    @param path A GtkTreePath pointing to the previous location of the deleted row.

    This should be called by models after a row has been removed. The location
    pointed to by path should be the location that the row previously was at.
    It may not be a valid location anymore.
 */
FALCON_FUNC TreeModel::row_deleted( VMARG )
{
    Item* i_path = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_row_deleted( (GtkTreeModel*)_obj, path );
}


/*#
    @method rows_reordered GtkTreeModel
    @brief Emits the "rows-reordered" signal on tree_model.
    @param path A GtkTreePath pointing to the tree node whose children have been reordered
    @param iter A valid GtkTreeIter pointing to the node whose children have been reordered, or NULL if the depth of path is 0.
    @param new_order an array of integers mapping the current position of each child to its old position before the re-ordering, i.e. new_order[newpos] = oldpos.

    This should be called by models when their rows have been reordered.
 */
FALCON_FUNC TreeModel::rows_reordered( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_iter = vm->param( 1 );
    Item* i_order = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isObject() || !IS_DERIVED( i_path, GtkTreePath )
        || !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_order || !i_order->isArray() )
        throw_inv_params( "GtkTreePath,GtkTreeIter,A" );
#endif
    GtkTreePath* path = GET_TREEPATH( *i_path );
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    CoreArray* order = i_order->asArray();
    const int cnt = order->length();
    MYSELF;
    GET_OBJ( self );
    if ( cnt == 0 ) // what to do?..
        gtk_tree_model_rows_reordered( (GtkTreeModel*)_obj, path, iter, NULL );
    else
    {
        gint* norder = (gint*) memAlloc( sizeof( gint ) * cnt );
        Item it;
        for ( int i = 0; i < cnt; ++i )
        {
            it = order->at( i );
#ifndef NO_PARAMETER_CHECK
            if ( !it.isInteger() )
            {
                memFree( norder );
                throw_inv_params( "I" );
            }
#endif
            norder[i] = it.asInteger();
        }
        gtk_tree_model_rows_reordered( (GtkTreeModel*)_obj, path, iter, norder );
        memFree( norder );
    }
}


} // Gtk
} // Falcon
