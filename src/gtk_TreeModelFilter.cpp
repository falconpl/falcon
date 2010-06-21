/**
 *  \file gtk_TreeModelFilter.cpp
 */

#include "gtk_TreeModelFilter.hpp"

//#include "gtk_TreeDragSource.hpp"
#include "gtk_TreeIter.hpp"
#include "gtk_TreeModel.hpp"
#include "gtk_TreePath.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeModelFilter::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeModelFilter = mod->addClass( "GtkTreeModelFilter", &TreeModelFilter::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_TreeModelFilter->getClassDef()->addInheritance( in );

    //c_TreeModelFilter->setWKS( true );
    c_TreeModelFilter->getClassDef()->factory( &TreeModelFilter::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_visible_func",           &TreeModelFilter::set_visible_func },
#if 0 // todo
    { "set_modify_func",            &TreeModelFilter::set_modify_func },
#endif
    { "set_visible_column",         &TreeModelFilter::set_visible_column },
    { "get_model",                  &TreeModelFilter::get_model },
    { "convert_child_iter_to_iter", &TreeModelFilter::convert_child_iter_to_iter },
    { "convert_iter_to_child_iter", &TreeModelFilter::convert_iter_to_child_iter },
    { "convert_child_path_to_path", &TreeModelFilter::convert_child_path_to_path },
    { "convert_path_to_child_path", &TreeModelFilter::convert_path_to_child_path },
    { "refilter",                   &TreeModelFilter::refilter },
    { "clear_cache",                &TreeModelFilter::clear_cache },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TreeModelFilter, meth->name, meth->cb );

    Gtk::TreeModel::clsInit( mod, c_TreeModelFilter );
    //Gtk::TreeDragSource::clsInit( mod, c_TreeModelFilter );
}


TreeModelFilter::TreeModelFilter( const Falcon::CoreClass* gen, const GtkTreeModelFilter* mdl )
    :
    Gtk::CoreGObject( gen, (GObject*) mdl )
{}


Falcon::CoreObject* TreeModelFilter::factory( const Falcon::CoreClass* gen, void* mdl, bool )
{
    return new TreeModelFilter( gen, (GtkTreeModelFilter*) mdl );
}


/*#
    @class GtkTreeModelFilter
    @brief A GtkTreeModel which hides parts of an underlying tree model
    @param child_model A GtkTreeModel.
    @param root A GtkTreePath or NULL.

    A GtkTreeModelFilter is a tree model which wraps another tree model, and can
    do the following things:

    - Filter specific rows, based on data from a "visible column", a column
    storing booleans indicating whether the row should be filtered or not, or
    based on the return value of a "visible function", which gets a model, iter
    and user_data and returns a boolean indicating whether the row should be
    filtered or not.

    - Modify the "appearance" of the model, using a modify function. This is
    extremely powerful and allows for just changing some values and also for
    creating a completely different model based on the given child model.

    - Set a different root node, also known as a "virtual root". You can pass in
    a GtkTreePath indicating the root node for the filter at construction time.

 */
FALCON_FUNC TreeModelFilter::init( VMARG )
{
    Item* i_mdl = vm->param( 0 );
    Item* i_root = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mdl || !i_mdl->isObject() || !Gtk::TreeModel::implementedBy( i_mdl )
        || !i_root || !( i_root->isNil() || ( i_root->isObject()
        && IS_DERIVED( i_root, GtkTreePath ) ) ) )
        throw_inv_params( "GtkTreeModel,[GtkTreePath]" );
#endif
    GtkTreeModel* mdl = GET_TREEMODEL( *i_mdl );
    GtkTreePath* root = i_root->isNil() ? NULL : GET_TREEPATH( i_root );
    MYSELF;
    self->setGObject( (GObject*) gtk_tree_model_filter_new( mdl, root ) );
}


/*#
    @method set_visible_func GtkTreeModelFilter
    @brief Sets the visible function used when filtering the filter to be func.
    @param func A GtkTreeModelFilterVisibleFunc, the visible function.
    @param data User data to pass to the visible function, or NULL.

    The function should return TRUE if the given row should be visible and
    FALSE otherwise.

    If the condition calculated by the function changes over time (e.g. because
    it depends on some global parameters), you must call
    gtk_tree_model_filter_refilter() to keep the visibility information of the
    model uptodate.

    Note that func is called whenever a row is inserted, when it may still be
    empty. The visible function should therefore take special care of empty rows,
    like in the example below.

    [...]
 */
FALCON_FUNC TreeModelFilter::set_visible_func( VMARG )
{
    Item* i_func = vm->param( 0 );
    Item* i_data = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_func || !i_func->isCallable()
        || !i_data )
        throw_inv_params( "C,[X]" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTreeModel* mdl = gtk_tree_model_filter_get_model( (GtkTreeModelFilter*)_obj );
    g_object_set_data_full( (GObject*) mdl,
                            "__tree_model_filter_visible_func__",
                            new GarbageLock( *i_func ),
                            &CoreGObject::release_lock );
    g_object_set_data_full( (GObject*) mdl,
                            "__tree_model_filter_visible_func_data__",
                            new GarbageLock( *i_data ),
                            &CoreGObject::release_lock );
    gtk_tree_model_filter_set_visible_func( (GtkTreeModelFilter*)_obj,
                                            &TreeModelFilter::exec_visible_func,
                                            (gpointer) vm,
                                            NULL );
}


gboolean TreeModelFilter::exec_visible_func( GtkTreeModel* mdl,
                                             GtkTreeIter* iter, gpointer _vm )
{
    GarbageLock* func_lock = (GarbageLock*) g_object_get_data( (GObject*) mdl,
                                        "__tree_model_filter_visible_func__" );
    GarbageLock* data_lock = (GarbageLock*) g_object_get_data( (GObject*) mdl,
                                        "__tree_model_filter_visible_func_data__" );
    assert( func_lock && data_lock );
    Item func = func_lock->item();
    Item data = func_lock->item();
    VMachine* vm = (VMachine*) _vm;
    vm->pushParam( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), iter ) );
    vm->pushParam( data );
    vm->callItem( func, 2 );
    Item it = vm->regA();
    if ( !it.isBoolean() )
    {
        printf( "TreeModelFilter::exec_visible_func: invalid callback (expected boolean)" );
        return FALSE;
    }
    return (gboolean) it.asBoolean();
}

#if 0 // todo
FALCON_FUNC TreeModelFilter::set_modify_func( VMARG );
#endif


/*#
    @method set_visible_column GtkTreeModelFilter
    @brief Sets column of the child_model to be the column where filter should look for visibility information.
    @param column An integer which is the column containing the visible information.

    columns should be a column of type G_TYPE_BOOLEAN, where TRUE means that a
    row is visible, and FALSE if not.
 */
FALCON_FUNC TreeModelFilter::set_visible_column( VMARG )
{
    Item* i_col = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_col || !i_col->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_filter_set_visible_column( (GtkTreeModelFilter*)_obj,
                                              i_col->asInteger() );
}


/*#
    @method get_model GtkTreeModelFilter
    @brief Returns a pointer to the child model of filter.
    @return a GtkTreeModel.
 */
FALCON_FUNC TreeModelFilter::get_model( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTreeModel* mdl = gtk_tree_model_filter_get_model( (GtkTreeModelFilter*)_obj );
    vm->retval( new Gtk::TreeModel( vm->findWKI( "GtkTreeModel" )->asClass(),
                                    mdl ) );
}


/*#
    @method convert_child_iter_to_iter GtkTreeModelFilter
    @brief Returns an iterator pointing to the row in filter that corresponds to the row pointed at by child_iter.
    @param child_iter A valid GtkTreeIter pointing to a row on the child model.
    @return a valid GtkTreeIter pointing to a visible row in child model.
    @raise ParamError if child_iter is invalid
 */
FALCON_FUNC TreeModelFilter::convert_child_iter_to_iter( VMARG )
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
    gboolean ret = gtk_tree_model_filter_convert_child_iter_to_iter(
                                                    (GtkTreeModelFilter*)_obj,
                                                    &iter, child );
    if ( ret )
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
    else
        throw_inv_params( "Valid GtkTreeIter" ); // todo: translate
}


/*#
    @method convert_iter_to_child_iter GtkTreeModelFilter
    @brief Returns an iterator pointing to the row pointed to by filter_iter.
    @param filter_iter A valid GtkTreeIter pointing to a row on filter.
    @return a GtkTreeIter.
 */
FALCON_FUNC TreeModelFilter::convert_iter_to_child_iter( VMARG )
{
    Item* i_filt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_filt || !i_filt->isObject() || !IS_DERIVED( i_filt, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* filt = GET_TREEITER( *i_filt );
    MYSELF;
    GET_OBJ( self );
    GtkTreeIter iter;
    gtk_tree_model_filter_convert_iter_to_child_iter( (GtkTreeModelFilter*)_obj,
                                                      &iter, filt );
    vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
}


/*#
    @method convert_child_path_to_path GtkTreeModelFilter
    @brief Converts child_path to a path relative to filter.
    @param child_path A GtkTreePath to convert.
    @return A GtkTreePath, or NULL.

    That is, child_path points to a path in the child model. The returned path
    will point to the same row in the filtered model. If child_path isn't a
    valid path on the child model or points to a row which is not visible in
    filter, then NULL is returned.
 */
FALCON_FUNC TreeModelFilter::convert_child_path_to_path( VMARG )
{
    Item* i_pth = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pth || !i_pth->isObject() || !IS_DERIVED( i_pth, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* path = GET_TREEPATH( i_pth );
    MYSELF;
    GET_OBJ( self );
    GtkTreePath* res =
    gtk_tree_model_filter_convert_child_path_to_path( (GtkTreeModelFilter*)_obj,
                                                      path );
    if ( res )
        vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                       res, true ) );
    else
        vm->retnil();
}


/*#
    @method convert_path_to_child_path GtkTreeModelFilter
    @brief Converts filter_path to a path on the child model of filter.

    That is, filter_path points to a location in filter. The returned path will
    point to the same location in the model not being filtered. If filter_path
    does not point to a location in the child model, NULL is returned.
 */
FALCON_FUNC TreeModelFilter::convert_path_to_child_path( VMARG )
{
    Item* i_pth = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pth || !i_pth->isObject() || !IS_DERIVED( i_pth, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* path = GET_TREEPATH( i_pth );
    MYSELF;
    GET_OBJ( self );
    GtkTreePath* res =
    gtk_tree_model_filter_convert_path_to_child_path( (GtkTreeModelFilter*)_obj,
                                                      path );
    if ( res )
        vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                       res, true ) );
    else
        vm->retnil();
}


/*#
    @method refilter GtkTreeModelFilter
    @brief Emits ::row_changed for each row in the child model, which causes the filter to re-evaluate whether a row is visible or not.
 */
FALCON_FUNC TreeModelFilter::refilter( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_filter_refilter( (GtkTreeModelFilter*)_obj );
}


/*#
    @method clear_cache GtkTreeModelFilter
    @brief This function should almost never be called.

    It clears the filter of any cached iterators that haven't been reffed with
    gtk_tree_model_ref_node(). This might be useful if the child model being
    filtered is static (and doesn't change often) and there has been a lot of
    unreffed access to nodes. As a side effect of this function, all unreffed
    iters will be invalid.
 */
FALCON_FUNC TreeModelFilter::clear_cache( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_filter_clear_cache( (GtkTreeModelFilter*)_obj );
}


} // Gtk
} // Falcon
