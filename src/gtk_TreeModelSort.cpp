/**
 *  \file gtk_TreeModelSort.cpp
 */

#include "gtk_TreeModelSort.hpp"

//#include "gtk_TreeDragSource.hpp"
#include "gtk_TreeIter.hpp"
#include "gtk_TreeModel.hpp"
#include "gtk_TreePath.hpp"
#include "gtk_TreeSortable.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeModelSort::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeModelSort = mod->addClass( "GtkTreeModelSort", &TreeModelSort::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_TreeModelSort->getClassDef()->addInheritance( in );

    //c_TreeModelSort->setWKS( true );
    c_TreeModelSort->getClassDef()->factory( &TreeModelSort::factory );

    Gtk::MethodTab methods[] =
    {
    { "get_model",                  &TreeModelSort::get_model },
    { "convert_child_path_to_path", &TreeModelSort::convert_child_path_to_path },
    { "convert_child_iter_to_iter", &TreeModelSort::convert_child_iter_to_iter },
    { "convert_path_to_child_path", &TreeModelSort::convert_path_to_child_path },
    { "convert_iter_to_child_iter", &TreeModelSort::convert_iter_to_child_iter },
    { "reset_default_sort_func",    &TreeModelSort::reset_default_sort_func },
    { "clear_cache",                &TreeModelSort::clear_cache },
    { "iter_is_valid",              &TreeModelSort::iter_is_valid },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TreeModelSort, meth->name, meth->cb );

    Gtk::TreeModel::clsInit( mod, c_TreeModelSort );
    //Gtk::TreeDragSource::clsInit( mod, c_TreeModelSort );
    Gtk::TreeSortable::clsInit( mod, c_TreeModelSort );
}


TreeModelSort::TreeModelSort( const Falcon::CoreClass* gen, const GtkTreeModelSort* mdl )
    :
    Gtk::CoreGObject( gen, (GObject*) mdl )
{}


Falcon::CoreObject* TreeModelSort::factory( const Falcon::CoreClass* gen, void* mdl, bool )
{
    return new TreeModelSort( gen, (GtkTreeModelSort*) mdl );
}


/*#
    @class GtkTreeModelSort
    @brief A GtkTreeModel which makes an underlying tree model sortable
    @param child_model A GtkTreeModel

    The GtkTreeModelSort is a model which implements the GtkTreeSortable
    interface. It does not hold any data itself, but rather is created with a
    child model and proxies its data. It has identical column types to this
    child model, and the changes in the child are propagated. The primary
    purpose of this model is to provide a way to sort a different model without
    modifying it. Note that the sort function used by GtkTreeModelSort is not
    guaranteed to be stable.

    [...]
 */
FALCON_FUNC TreeModelSort::init( VMARG )
{
    Item* i_mdl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mdl || !i_mdl->isObject() || !Gtk::TreeModel::implementedBy( i_mdl ) )
        throw_inv_params( "GtkTreeModel" );
#endif
    GtkTreeModel* mdl = GET_TREEMODEL( *i_mdl );
    MYSELF;
    self->setGObject( (GObject*) gtk_tree_model_sort_new_with_model( mdl ) );
}


/*#
    @method get_model GtkTreeModelSort
    @brief Returns the model the GtkTreeModelSort is sorting.
    @return the "child model" being sorted (GtkTreeModel).
 */
FALCON_FUNC TreeModelSort::get_model( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( new Gtk::TreeModel( vm->findWKI( "GtkTreeModel" )->asClass(),
                    gtk_tree_model_sort_get_model( (GtkTreeModelSort*)_obj ) ) );
}


/*#
    @method convert_child_path_to_path GtkTreeModelSort
    @brief Converts child_path to a path relative to the tree-model-sort.
    @param child_path A GtkTreePath to convert
    @return A newly allocated GtkTreePath, or NULL

    That is, child_path points to a path in the child model. The returned path
    will point to the same row in the sorted model. If child_path isn't a valid
    path on the child model, then NULL is returned.
 */
FALCON_FUNC TreeModelSort::convert_child_path_to_path( VMARG )
{
    Item* i_pth = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pth || !i_pth->isObject() || !IS_DERIVED( i_pth, GtkTreepath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* pth = GET_TREEPATH( *i_pth );
    MYSELF;
    GET_OBJ( self );
    GtkTreePath* res = gtk_tree_model_sort_convert_child_path_to_path(
                                                (GtkTreeModelSort*)_obj, pth );
    if ( res )
        vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                       res,
                                       true ) );
    else
        vm->retnil();
}


/*#
    @method convert_child_iter_to_iter GtkTreeModelSort
    @brief Returns an iterator pointing to the row in tree_model_sort that corresponds to the row pointed at by child_iter.
    @param child_iter A valid GtkTreeIter pointing to a row on the child model
    @return a valid GtkTreeIter to a visible row in the child model, or Nil
 */
FALCON_FUNC TreeModelSort::convert_child_iter_to_iter( VMARG )
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
#if GTK_CHECK_VERSION( 2, 14, 0 )
    gboolean ret =
#endif
    gtk_tree_model_sort_convert_child_iter_to_iter( (GtkTreeModelSort*)_obj,
                                                    &iter, child );
#if GTK_CHECK_VERSION( 2, 14, 0 )
    if ( ret )
#else
    if ( gtk_tree_model_sort_iter_is_valid( (GtkTreeModelSort*)_obj, &iter ) )
#endif
        vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &iter ) );
    else
        vm->retnil();
}


/*#
    @method convert_path_to_child_path GtkTreeModelSort
    @brief Converts sorted_path to a path on the child model of tree_model_sort.
    @param sorted_path A GtkTreePath to convert
    @return A GtkTreePath, or NULL

    That is, sorted_path points to a location in tree_model_sort. The returned
    path will point to the same location in the model not being sorted. If
    sorted_path does not point to a location in the child model, NULL is returned.
 */
FALCON_FUNC TreeModelSort::convert_path_to_child_path( VMARG )
{
    Item* i_pth = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pth || !i_pth->isObject() || !IS_DERIVED( i_pth, GtkTreePath ) )
        throw_inv_params( "GtkTreePath" );
#endif
    GtkTreePath* pth = GET_TREEPATH( *i_pth );
    MYSELF;
    GET_OBJ( self );
    GtkTreePath* res = gtk_tree_model_sort_convert_path_to_child_path(
                                                (GtkTreeModelSort*)_obj, pth );
    if ( res )
        vm->retval( new Gtk::TreePath( vm->findWKI( "GtkTreePath" )->asClass(),
                                       res,
                                       true ) );
    else
        vm->retnil();
}


/*#
    @method convert_iter_to_child_iter GtkTreeModelSort
    @brief Returns an iterator pointing to the row pointed to by sorted_iter.
    @param sorted_iter A valid GtkTreeIter pointing to a row on tree_model_sort.
    @return a GtkTreeIter
 */
FALCON_FUNC TreeModelSort::convert_iter_to_child_iter( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    GtkTreeIter res;
    gtk_tree_model_sort_convert_iter_to_child_iter( (GtkTreeModelSort*)_obj,
                                                    &res, iter );
    vm->retval( new Gtk::TreeIter( vm->findWKI( "GtkTreeIter" )->asClass(), &res ) );
}


/*#
    @method reset_default_sort_func GtkTreeModelSort
    @brief This resets the default sort function to be in the 'unsorted' state.
    That is, it is in the same order as the child model. It will re-sort the
    model to be in the same order as the child model only if the GtkTreeModelSort
    is in 'unsorted' state.
 */
FALCON_FUNC TreeModelSort::reset_default_sort_func( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_sort_reset_default_sort_func( (GtkTreeModelSort*)_obj );
}


/*#
    @method clear_cache GtkTreeModelSort
    @brief This function should almost never be called.

    It clears the tree_model_sort of any cached iterators that haven't been
    reffed with gtk_tree_model_ref_node(). This might be useful if the child
    model being sorted is static (and doesn't change often) and there has been
    a lot of unreffed access to nodes. As a side effect of this function, all
    unreffed iters will be invalid.
 */
FALCON_FUNC TreeModelSort::clear_cache( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_tree_model_sort_clear_cache( (GtkTreeModelSort*)_obj );
}


/*#
    @method iter_is_valid GtkTreeModelSort
    @brief Checks if the given iter is a valid iter for this GtkTreeModelSort.
    @param iter A GtkTreeIter.
    @return TRUE if the iter is valid, FALSE if the iter is invalid.

    Warning: This function is slow. Only use it for debugging and/or testing purposes.
 */
FALCON_FUNC TreeModelSort::iter_is_valid( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = GET_TREEITER( *i_iter );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tree_model_sort_iter_is_valid( (GtkTreeModelSort*)_obj, iter ) );
}


} // Gtk
} // Falcon
