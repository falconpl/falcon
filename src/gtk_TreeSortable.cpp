/**
 *  \file gtk_TreeSortable.cpp
 */

#include "gtk_TreeSortable.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeSortable::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeSortable = mod->addClass( "%GtkTreeSortable" );

    c_TreeSortable->setWKS( true );
    c_TreeSortable->getClassDef()->factory( &TreeSortable::factory );

    TreeSortable::clsInit( mod, c_TreeSortable );
}


/**
 *  \brief interface loader
 */
void TreeSortable::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "signal_sort_column_changed",&TreeSortable::signal_sort_column_changed },
    { "sort_column_changed",    &TreeSortable::sort_column_changed },
    { "get_sort_column_id",     &TreeSortable::get_sort_column_id },
    { "set_sort_column_id",     &TreeSortable::set_sort_column_id },
#if 0 // todo
    { "set_sort_func",          &TreeSortable::set_sort_func },
    { "set_default_sort_func",  &TreeSortable::set_default_sort_func },
#endif
    { "has_default_sort_func",  &TreeSortable::has_default_sort_func },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


TreeSortable::TreeSortable( const Falcon::CoreClass* gen, const GtkTreeSortable* model )
    :
    Gtk::CoreGObject( gen, (GObject*) model )
{}


Falcon::CoreObject* TreeSortable::factory( const Falcon::CoreClass* gen, void* model, bool )
{
    return new TreeSortable( gen, (GtkTreeSortable*) model );
}


/*#
    @class GtkTreeSortable
    @brief The interface for sortable models used by GtkTreeView

    GtkTreeSortable is an interface to be implemented by tree models which
    support sorting. The GtkTreeView uses the methods provided by this interface
    to sort the model.
 */


/*#
    @method signal_sort_column_changed GtkTreeSortable
    @brief The ::sort-column-changed signal is emitted when the sort column or sort order of sortable is changed.

    The signal is emitted before the contents of sortable are resorted.
 */
FALCON_FUNC TreeSortable::signal_sort_column_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "sort_column_changed",
                             (void*) &TreeSortable::on_sort_column_changed, vm );
}


void TreeSortable::on_sort_column_changed( GtkTreeSortable* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "sort_column_changed",
        "on_sort_column_changed", (VMachine*)_vm );
}


/*#
    @method sort_column_changed GtkTreeSortable
    @brief Emits a "sort-column-changed" signal on sortable.
 */
FALCON_FUNC TreeSortable::sort_column_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tree_sortable_sort_column_changed( (GtkTreeSortable*)_obj );
}


/*#
    @method get_sort_column_id GtkTreeSortable
    @brief Returns the sort column id and order with the current sort column and the order.
    @return an array ( sort column id, GtkSortType ).

    If the sort column ID is not set, then (-2, 0) is returned.
    If the sort column ID is set to -1 indicating the default sort function is
    to be used this method returns (-1, 0).
 */
FALCON_FUNC TreeSortable::get_sort_column_id( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gint id;
    GtkSortType order;
    gboolean ret = gtk_tree_sortable_get_sort_column_id( (GtkTreeSortable*)_obj,
                                                         &id,
                                                         &order );
    CoreArray* arr = new CoreArray( 2 );
    if ( ret )
    {
        arr->append( id );
        arr->append( (int64) order );
    }
    else
    {
        switch ( id )
        {
        case GTK_TREE_SORTABLE_DEFAULT_SORT_COLUMN_ID:
            arr->append( -1 );
            break;
        case GTK_TREE_SORTABLE_UNSORTED_SORT_COLUMN_ID:
            arr->append( -2 );
            break;
        default: // not reached
            return;
        }
        arr->append( 0 );
    }
    vm->retval( arr );
}


/*#
    @method set_sort_column_id GtkTreeSortable
    @brief Sets the current sort column to be sort_column_id.
    @param sort_column_id the sort column id to set
    @param order The sort order of the column (GtkSortType)

    The sortable will resort itself to reflect this change, after emitting a
    "sort-column-changed" signal. sortable may either be a regular column id,
    or one of the following special values:

    - GTK_TREE_SORTABLE_DEFAULT_SORT_COLUMN_ID: the default sort function will
    be used, if it is set

    - GTK_TREE_SORTABLE_UNSORTED_SORT_COLUMN_ID: no sorting will occur
 */
FALCON_FUNC TreeSortable::set_sort_column_id( VMARG )
{
    Item* i_id = vm->param( 0 );
    Item* i_ord = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || !i_id->isInteger()
        || !i_ord || !i_ord->isInteger() )
        throw_inv_params( "I,GtkSortType" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tree_sortable_set_sort_column_id( (GtkTreeSortable*)_obj,
                                          i_id->asInteger(),
                                          (GtkSortType) i_ord->asInteger() );
}


#if 0 // todo
FALCON_FUNC TreeSortable::set_sort_func( VMARG );
FALCON_FUNC TreeSortable::set_default_sort_func( VMARG );
#endif


/*#
    @method has_default_sort_func GtkTreeSortable
    @brief Returns TRUE if the model has a default sort function.
    @return TRUE, if the model has a default sort function

    This is used primarily by GtkTreeViewColumns in order to determine if a
    model can go back to the default state, or not.
 */
FALCON_FUNC TreeSortable::has_default_sort_func( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tree_sortable_has_default_sort_func( (GtkTreeSortable*)_obj ) );
}


} // Gtk
} // Falcon
