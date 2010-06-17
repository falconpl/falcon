/**
 *  \file gtk_TreeStore.cpp
 */

#include "gtk_TreeStore.hpp"

#include "gtk_Buildable.hpp"
#include "gtk_TreeIter.hpp"
#include "gtk_TreeModel.hpp"
//#include "gtk_TreeDragDest.hpp"
//#include "gtk_TreeDragSource.hpp"
#include "gtk_TreeSortable.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TreeStore::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TreeStore = mod->addClass( "GtkTreeStore", &TreeStore::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_TreeStore->getClassDef()->addInheritance( in );

    //c_TreeStore->setWKS( true );
    c_TreeStore->getClassDef()->factory( &TreeStore::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_column_types",       &TreeStore::set_column_types },
    { "set_value",              &TreeStore::set_value },
    { "set",                    &TreeStore::set },
#if 0 // unused
    { "set_valist",             &TreeStore::set_valist },
    { "set_valuesv",            &TreeStore::set_valuesv },
#endif
    { "remove",                 &TreeStore::remove },
    { "insert",                 &TreeStore::insert },
    { "insert_before",          &TreeStore::insert_before },
    { "insert_after",           &TreeStore::insert_after },
    { "insert_with_values",     &TreeStore::insert_with_values },
#if 0 // unused
    { "insert_with_valuesv",    &TreeStore::insert_with_valuesv },
#endif
    { "prepend",                &TreeStore::prepend },
    { "append",                 &TreeStore::append },
    { "is_ancestor",            &TreeStore::is_ancestor },
    { "iter_depth",             &TreeStore::iter_depth },
    { "clear",                  &TreeStore::clear },
    { "iter_is_valid",          &TreeStore::iter_is_valid },
    { "reorder",                &TreeStore::reorder },
    { "swap",                   &TreeStore::swap },
    { "move_before",            &TreeStore::move_before },
    { "move_after",             &TreeStore::move_after },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TreeStore, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_TreeStore );
    Gtk::TreeModel::clsInit( mod, c_TreeStore );
    //Gtk::TreeDragDest::clsInit( mod, c_TreeStore );
    //Gtk::TreeDragSource::clsInit( mod, c_TreeStore );
    Gtk::TreeSortable::clsInit( mod, c_TreeStore );
}


TreeStore::TreeStore( const Falcon::CoreClass* gen, const GtkTreeStore* store )
    :
    Gtk::CoreGObject( gen, (GObject*) store )
{}


Falcon::CoreObject* TreeStore::factory( const Falcon::CoreClass* gen, void* store, bool )
{
    return new TreeStore( gen, (GtkTreeStore*) store );
}


/*#
    @class GtkTreeStore
    @brief A tree-like data structure that can be used with the GtkTreeView
    @param types an array of all GType types for the columns, from first to last

    The GtkTreeStore object is a list model for use with a GtkTreeView widget.
    It implements the GtkTreeModel interface, and consequentialy, can use all of
    the methods available there. It also implements the GtkTreeSortable
    interface so it can be sorted by the view. Finally, it also implements the
    tree drag and drop interfaces.
 */
FALCON_FUNC TreeStore::init( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
    CoreArray* arr = i_arr->asArray();
    gint ncol = arr->length();
    GtkTreeStore* tree;
    if ( ncol == 0 )
        tree = gtk_tree_store_newv( 0, NULL );
    else
    {
        GType* types = (GType*) memAlloc( sizeof( GType ) * ncol );
        Item it;
        for ( int i = 0; i < ncol; ++i )
        {
            it = arr->at( i );
#ifndef NO_PARAMETER_CHECK
            if ( !it.isInteger() )
            {
                memFree( types );
                throw_inv_params( "GType" );
            }
#endif
            types[i] = it.asInteger();
        }
        tree = gtk_tree_store_newv( ncol, types );
        memFree( types );
    }
    MYSELF;
    self->setGObject( (GObject*) tree );
}


/*#
    @method set_column_types GtkTreeStore
    @brief Sets the column types.
    @param types an array of GType

    This function is meant primarily for GObjects that inherit from GtkTreeStore,
    and should only be used when constructing a new GtkTreeStore.

    It will not function after a row has been added, or a method on the
    GtkTreeModel interface is called.
 */
FALCON_FUNC TreeStore::set_column_types( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
    CoreArray* arr = i_arr->asArray();
    gint ncol = arr->length();
    MYSELF;
    GET_OBJ( self );
    if ( ncol == 0 )
        gtk_tree_store_set_column_types( (GtkTreeStore*)_obj, 0, NULL );
    else
    {
        GType* types = (GType*) memAlloc( sizeof( GType ) * ncol );
        Item it;
        for ( int i = 0; i < ncol; ++i )
        {
            it = arr->at( i );
#ifndef NO_PARAMETER_CHECK
            if ( !it.isInteger() )
            {
                memFree( types );
                throw_inv_params( "GType" );
            }
#endif
            types[i] = it.asInteger();
        }
        gtk_tree_store_set_column_types( (GtkTreeStore*)_obj, ncol, types );
        memFree( types );
    }
}


/*#
    @method set_value GtkTreeStore
    @brief Sets the data in the cell specified by iter and column.
    @param iter A valid GtkTreeIter for the row being modified
    @param column column number to modify
    @param value new value for the cell

    The type of value must be convertible to the type of the column.
 */
FALCON_FUNC TreeStore::set_value( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_col = vm->param( 1 );
    Item* i_val = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_col || !i_col->isInteger()
        || !i_val )
        throw_inv_params( "GtkTreeIter,I,X" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GValue val;
    switch ( i_val->type() )
    {
    case FLC_ITEM_NIL:
        g_value_init( &val, G_TYPE_NONE );
        break;
    case FLC_ITEM_INT:
        g_value_init( &val, G_TYPE_INT64 );
        g_value_set_int64( &val, i_val->asInteger() );
        break;
    case FLC_ITEM_BOOL:
        g_value_init( &val, G_TYPE_BOOLEAN );
        g_value_set_boolean( &val, (gboolean) i_val->asBoolean() );
        break;
    case FLC_ITEM_NUM:
        g_value_init( &val, G_TYPE_DOUBLE );
        g_value_set_double( &val, i_val->asNumeric() );
        break;
    case FLC_ITEM_STRING:
    {
        AutoCString tmp( i_val->asString() );
        g_value_init( &val, G_TYPE_STRING );
        g_value_set_string( &val, tmp.c_str() );
        break;
    }
    case FLC_ITEM_OBJECT:
    {
#ifndef NO_PARAMETER_CHECK
        if ( !IS_DERIVED( i_val, GObject ) )
            throw_inv_params( "GObject" );
#endif
        GObject* obj = dyncast<Gtk::CoreGObject*>( i_val->asObjectSafe() )->getGObject();
        g_value_init( &val, G_TYPE_OBJECT );
        g_value_set_object( &val, obj );
        break;
    }
    default:
        throw_inv_params( "Not implemented" );
    }
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_set_value( (GtkTreeStore*)_obj,
                              iter, i_col->asInteger(), &val );
}


/*#
    @method set GtkTreeStore
    @brief Sets the value of one or more cells in the row referenced by iter.
    @param iter row iterator (GtkTreeIter)
    @param values an array of pairs [ column index, column value, ... ]

    The variable argument list should contain integer column numbers, each
    column number followed by the value to be set. For example, to set column 0
    with type G_TYPE_STRING to "Foo", you would write
    gtk_tree_store_set ( iter, [ 0, "Foo" ] ).
 */
FALCON_FUNC TreeStore::set( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_pairs = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_pairs || !i_pairs->isArray() )
        throw_inv_params( "GtkTreeIter,A" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    CoreArray* pairs = i_pairs->asArray();
    const int n = pairs->length();
#ifndef NO_PARAMETER_CHECK
    if ( ( n % 2 ) != 0 )
        throw_inv_params( "pairs of column number and value" ); // todo: translate
#endif
    MYSELF;
    GET_OBJ( self );
    if ( n == 0 )
        gtk_tree_store_set( (GtkTreeStore*)_obj, iter, -1 );
    else
    {
        const int npairs = n / 2;
        gint* indexes = (gint*) memAlloc( sizeof( gint ) * npairs );
        GValue* values = (GValue*) memAlloc( sizeof( GValue ) * npairs );
        Item it;
        for ( int i = 0; i < n; i += 2 )
        {
            // set the index
            it = pairs->at( i );
#ifndef NO_PARAMETER_CHECK
            if ( !it.isInteger() )
            {
                memFree( indexes );
                memFree( values );
                throw_inv_params( "I" );
            }
#endif
            indexes[i] = it.asInteger();
            // set the value
            it = pairs->at( i + 1 );
            switch ( it.type() )
            {
            case FLC_ITEM_NIL:
                g_value_init( &values[i], G_TYPE_NONE );
                break;
            case FLC_ITEM_INT:
                g_value_init( &values[i], G_TYPE_INT64 );
                g_value_set_int64( &values[i], it.asInteger() );
                break;
            case FLC_ITEM_BOOL:
                g_value_init( &values[i], G_TYPE_BOOLEAN );
                g_value_set_boolean( &values[i], (gboolean) it.asBoolean() );
                break;
            case FLC_ITEM_NUM:
                g_value_init( &values[i], G_TYPE_DOUBLE );
                g_value_set_double( &values[i], it.asNumeric() );
                break;
            case FLC_ITEM_STRING:
            {
                AutoCString tmp( it.asString() );
                g_value_init( &values[i], G_TYPE_STRING );
                g_value_set_string( &values[i], tmp.c_str() );
                break;
            }
            case FLC_ITEM_OBJECT:
            {
#ifndef NO_PARAMETER_CHECK
                if ( !IS_DERIVED( &it, GObject ) )
                {
                    memFree( indexes );
                    memFree( values );
                    throw_inv_params( "GObject" );
                }
#endif
                GObject* obj = dyncast<Gtk::CoreGObject*>( it.asObjectSafe() )->getGObject();
                g_value_init( &values[i], G_TYPE_OBJECT );
                g_value_set_object( &values[i], obj );
                break;
            }
            default:
                memFree( indexes );
                memFree( values );
                throw_inv_params( "Not implemented" );
            }
        }
        gtk_tree_store_set_valuesv( (GtkTreeStore*)_obj,
                                    iter, indexes, values, npairs );
        memFree( indexes );
        memFree( values );
    }
}


#if 0 // unused
FALCON_FUNC TreeStore::set_valist( VMARG );
FALCON_FUNC TreeStore::set_valuesv( VMARG );
#endif


/*#
    @method remove GtkTreeStore
    @brief Removes iter from tree_store.
    @param iter A valid GtkTreeIter
    @return TRUE if iter is still valid, FALSE if not.

    After being removed, iter is set to the next valid row at that level, or
    invalidated if it previously pointed to the last one.
 */
FALCON_FUNC TreeStore::remove( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tree_store_remove( (GtkTreeStore*)_obj, iter ) );
}


/*#
    @method insert GtkTreeStore
    @brief Creates a new row at position.
    @param iter An unset GtkTreeIter to set to the new row
    @param parent A valid GtkTreeIter, or NULL.
    @param position position to insert the new row

    If parent is non-NULL, then the row will be made a child of parent.
    Otherwise, the row will be created at the toplevel. If position is larger
    than the number of rows at that level, then the new row will be inserted to
    the end of the list. iter will be changed to point to this new row. The row
    will be empty after this function is called. To fill in values, you need to
    call gtk_tree_store_set() or gtk_tree_store_set_value().
 */
FALCON_FUNC TreeStore::insert( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_par = vm->param( 1 );
    Item* i_pos = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) )
        || !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter],I" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* parent = i_par->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_par->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_insert( (GtkTreeStore*)_obj, iter, parent, i_pos->asInteger() );
}


/*#
    @method insert_before GtkTreeStore
    @brief Inserts a new row before sibling.
    @param iter An unset GtkTreeIter to set to the new row
    @param parent A valid GtkTreeIter, or NULL.
    @param sibling A valid GtkTreeIter, or NULL.

    If sibling is NULL, then the row will be appended to parent 's children. If
    parent and sibling are NULL, then the row will be appended to the toplevel.
    If both sibling and parent are set, then parent must be the parent of sibling.
    When sibling is set, parent is optional.

    iter will be changed to point to this new row. The row will be empty after
    this function is called. To fill in values, you need to call
    gtk_tree_store_set() or gtk_tree_store_set_value().
 */
FALCON_FUNC TreeStore::insert_before( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_par = vm->param( 1 );
    Item* i_sib = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) )
        || !i_sib || !( i_sib->isNil() || ( i_sib->isObject()
        && IS_DERIVED( i_sib, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter],[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* par = i_par->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_par->asObjectSafe() )->getTreeIter();
    GtkTreeIter* sib = i_sib->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_sib->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_insert_before( (GtkTreeStore*)_obj, iter, par, sib );
}


/*#
    @method insert_after GtkTreeStore
    @brief Inserts a new row after sibling.
    @param iter An unset GtkTreeIter to set to the new row
    @param parent A valid GtkTreeIter, or NULL.
    @param sibling A valid GtkTreeIter, or NULL.

    If sibling is NULL, then the row will be prepended to parent 's children. If
    parent and sibling are NULL, then the row will be prepended to the toplevel.
    If both sibling and parent are set, then parent must be the parent of sibling.
    When sibling is set, parent is optional.

    iter will be changed to point to this new row. The row will be empty after
    this function is called. To fill in values, you need to call
    gtk_tree_store_set() or gtk_tree_store_set_value().
 */
FALCON_FUNC TreeStore::insert_after( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_par = vm->param( 1 );
    Item* i_sib = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) )
        || !i_sib || !( i_sib->isNil() || ( i_sib->isObject()
        && IS_DERIVED( i_sib, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter],[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* par = i_par->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_par->asObjectSafe() )->getTreeIter();
    GtkTreeIter* sib = i_sib->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_sib->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_insert_after( (GtkTreeStore*)_obj, iter, par, sib );
}


/*#
    @method insert_with_values GtkTreeStore
    @brief Creates a new row at position.
    @param iter An unset GtkTreeIter to set to the new row, or NULL.
    @param parent A valid GtkTreeIter, or NULL.
    @param pos position to insert the new row
    @param values an array of pairs [ column index, column value, ... ]

    iter will be changed to point to this new row. If position is larger than
    the number of rows on the list, then the new row will be appended to the
    list. The row will be filled with the values given to this function.
 */
FALCON_FUNC TreeStore::insert_with_values( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_par = vm->param( 1 );
    Item* i_pos = vm->param( 2 );
    Item* i_pairs = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !( i_iter->isNil() || ( i_iter->isObject()
        && IS_DERIVED( i_iter, GtkTreeIter ) ) )
        || !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) )
        || !i_pos || !i_pos->isInteger()
        || !i_pairs || !i_pairs->isArray() )
        throw_inv_params( "[GtkTreeIter],[GtkTreeIter],I,A" );
#endif
    GtkTreeIter* iter = i_iter->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* par = i_par->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_par->asObjectSafe() )->getTreeIter();
    CoreArray* pairs = i_pairs->asArray();
    const int n = pairs->length();
#ifndef NO_PARAMETER_CHECK
    if ( ( n % 2 ) != 0 )
        throw_inv_params( "pairs of column number and value" ); // todo: translate
#endif
    MYSELF;
    GET_OBJ( self );
    if ( n == 0 )
        gtk_tree_store_insert_with_values( (GtkTreeStore*)_obj,
                                           iter, par, i_pos->asInteger(), -1 );
    else
    {
        const int npairs = n / 2;
        gint* indexes = (gint*) memAlloc( sizeof( gint ) * npairs );
        GValue* values = (GValue*) memAlloc( sizeof( GValue ) * npairs );
        Item it;
        for ( int i = 0; i < n; i += 2 )
        {
            // set the index
            it = pairs->at( i );
#ifndef NO_PARAMETER_CHECK
            if ( !it.isInteger() )
            {
                memFree( indexes );
                memFree( values );
                throw_inv_params( "I" );
            }
#endif
            indexes[i] = it.asInteger();
            // set the value
            it = pairs->at( i + 1 );
            switch ( it.type() )
            {
            case FLC_ITEM_NIL:
                g_value_init( &values[i], G_TYPE_NONE );
                break;
            case FLC_ITEM_INT:
                g_value_init( &values[i], G_TYPE_INT64 );
                g_value_set_int64( &values[i], it.asInteger() );
                break;
            case FLC_ITEM_BOOL:
                g_value_init( &values[i], G_TYPE_BOOLEAN );
                g_value_set_boolean( &values[i], (gboolean) it.asBoolean() );
                break;
            case FLC_ITEM_NUM:
                g_value_init( &values[i], G_TYPE_DOUBLE );
                g_value_set_double( &values[i], it.asNumeric() );
                break;
            case FLC_ITEM_STRING:
            {
                AutoCString tmp( it.asString() );
                g_value_init( &values[i], G_TYPE_STRING );
                g_value_set_string( &values[i], tmp.c_str() );
                break;
            }
            case FLC_ITEM_OBJECT:
            {
#ifndef NO_PARAMETER_CHECK
                if ( !IS_DERIVED( &it, GObject ) )
                {
                    memFree( indexes );
                    memFree( values );
                    throw_inv_params( "GObject" );
                }
#endif
                GObject* obj = dyncast<Gtk::CoreGObject*>( it.asObjectSafe() )->getGObject();
                g_value_init( &values[i], G_TYPE_OBJECT );
                g_value_set_object( &values[i], obj );
                break;
            }
            default:
                memFree( indexes );
                memFree( values );
                throw_inv_params( "Not implemented" );
            }
        }
        gtk_tree_store_insert_with_valuesv( (GtkTreeStore*)_obj,
                        iter, par, i_pos->asInteger(), indexes, values, npairs );
        memFree( indexes );
        memFree( values );
    }
}


#if 0 // unused
FALCON_FUNC TreeStore::insert_with_valuesv( VMARG );
#endif


/*#
    @method prepend GtkTreeStore
    @brief Prepends a new row to tree_store.
    @param iter An unset GtkTreeIter to set to the prepended row.
    @param parent A valid GtkTreeIter, or NULL.

    If parent is non-NULL, then it will prepend the new row before the first
    child of parent, otherwise it will prepend a row to the top level. iter will
    be changed to point to this new row. The row will be empty after this
    function is called. To fill in values, you need to call
    gtk_tree_store_set() or gtk_tree_store_set_value().
 */
FALCON_FUNC TreeStore::prepend( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_par = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* par = i_par->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_par->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_prepend( (GtkTreeStore*)_obj, iter, par );
}


/*#
    @method append GtkTreeStore
    @brief Appends a new row to tree_store.
    @param iter An unset GtkTreeIter to set to the appended row
    @param parent A valid GtkTreeIter, or NULL.

    If parent is non-NULL, then it will append the new row after the last child
    of parent, otherwise it will append a row to the top level. iter will be
    changed to point to this new row. The row will be empty after this function
    is called. To fill in values, you need to call gtk_tree_store_set() or
    gtk_tree_store_set_value().
 */
FALCON_FUNC TreeStore::append( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_par = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* par = i_par->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_par->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_append( (GtkTreeStore*)_obj, iter, par );
}


/*#
    @method is_ancestor GtkTreeStore
    @brief Returns TRUE if iter is an ancestor of descendant.
    @param iter A valid GtkTreeIter
    @param descendant A valid GtkTreeIter
    @return TRUE, if iter is an ancestor of descendant

    That is, iter is the parent (or grandparent or great-grandparent) of descendant.
 */
FALCON_FUNC TreeStore::is_ancestor( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_desc = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_desc || !i_desc->isObject() || !IS_DERIVED( i_desc, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter,GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* desc = dyncast<Gtk::TreeIter*>( i_desc->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tree_store_is_ancestor( (GtkTreeStore*)_obj, iter, desc ) );
}


/*#
    @method iter_depth GtkTreeStore
    @brief Returns the depth of iter.
    @param iter A valid GtkTreeIter
    @return The depth of iter

    This will be 0 for anything on the root level, 1 for anything down a level, etc.
 */
FALCON_FUNC TreeStore::iter_depth( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_tree_store_iter_depth( (GtkTreeStore*)_obj, iter ) );
}


/*#
    @method clear GtkTreeStore
    @brief Removes all rows from tree_store
 */
FALCON_FUNC TreeStore::clear( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_clear( (GtkTreeStore*)_obj );
}


/*#
    @method iter_is_valid GtkTreeStore
    @brief Checks if the given iter is a valid iter for this GtkTreeStore.
    @param iter A GtkTreeIter.
    @return TRUE if the iter is valid, FALSE if the iter is invalid.

    Warning: This function is slow. Only use it for debugging and/or testing purposes.
 */
FALCON_FUNC TreeStore::iter_is_valid( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tree_store_iter_is_valid( (GtkTreeStore*)_obj, iter ) );
}


/*#
    @method reorder GtkTreeStore
    @brief Reorders the children of parent in tree_store to follow the order indicated by new_order.
    @param parent A GtkTreeIter.
    @param new_order an array of integers mapping the new position of each child to its old position before the re-ordering, i.e. new_order[newpos] = oldpos.

    Note that this function only works with unsorted stores.
 */
FALCON_FUNC TreeStore::reorder( VMARG )
{
    Item* i_par = vm->param( 0 );
    Item* i_arr = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_par || !( i_par->isNil() || ( i_par->isObject()
        && IS_DERIVED( i_par, GtkTreeIter ) ) )
        || !i_arr || !i_arr->isArray() )
        throw_inv_params( "GtkTreeIter,A" );
#endif
    GtkTreeIter* par = i_par->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_par->asObjectSafe() )->getTreeIter();
    CoreArray* arr = i_arr->asArray();
    const int n = arr->length();
#ifndef NO_PARAMETER_CHECK
    if ( n == 0 )
        throw_inv_params( "Non-empty array" ); // todo: translate
#endif
    gint* order = (gint*) memAlloc( sizeof( gint ) * n );
    Item it;
    for ( int i = 0; i < n; ++i )
    {
        it = arr->at( i );
#ifndef NO_PARAMETER_CHECK
        if ( !it.isInteger() )
        {
            memFree( order );
            throw_inv_params( "I" );
        }
#endif
        order[i] = it.asInteger();
    }
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_reorder( (GtkTreeStore*)_obj, par, order );
    memFree( order );
}


/*#
    @method swap GtkTreeStore
    @brief Swaps a and b in the same level of tree_store.
    @param a A GtkTreeIter.
    @param b Another GtkTreeIter.

    Note that this function only works with unsorted stores.
 */
FALCON_FUNC TreeStore::swap( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_iter2 = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_iter2 || !i_iter2->isObject() || !IS_DERIVED( i_iter2, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter,GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* iter2 = dyncast<Gtk::TreeIter*>( i_iter2->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_swap( (GtkTreeStore*)_obj, iter, iter2 );
}


/*#
    @method move_before GtkTreeStore
    @brief Moves iter in tree_store to the position before position.
    @param iter A GtkTreeIter.
    @param position A GtkTreeIter or NULL.

    iter and position should be in the same level. Note that this function only
    works with unsorted stores. If position is NULL, iter will be moved to the
    end of the level.
 */
FALCON_FUNC TreeStore::move_before( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_iter2 = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_iter2 || !( i_iter2->isNil() || ( i_iter2->isObject()
        && IS_DERIVED( i_iter2, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* iter2 = i_iter2->isNil() ? NULL
            : dyncast<Gtk::TreeIter*>( i_iter2->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_move_before( (GtkTreeStore*)_obj, iter, iter2 );
}


/*#
    @method move_after GtkTreeStore
    @brief Moves iter in tree_store to the position after position.
    @param iter A GtkTreeIter.
    @param position A GtkTreeIter or NULL.

    iter and position should be in the same level. Note that this function only
    works with unsorted stores. If position is NULL, iter will be moved to the
    start of the level.
 */
FALCON_FUNC TreeStore::move_after( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_iter2 = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_iter2 || !( i_iter2->isNil() || ( i_iter2->isObject()
        && IS_DERIVED( i_iter2, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* iter2 = i_iter2->isNil() ? NULL
            : dyncast<Gtk::TreeIter*>( i_iter2->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_tree_store_move_after( (GtkTreeStore*)_obj, iter, iter2 );
}


} // Gtk
} // Falcon
