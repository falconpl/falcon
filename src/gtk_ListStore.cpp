/**
 *  \file gtk_ListStore.cpp
 */

#include "gtk_ListStore.hpp"

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
void ListStore::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ListStore = mod->addClass( "GtkListStore", &ListStore::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_ListStore->getClassDef()->addInheritance( in );

    //c_ListStore->setWKS( true );
    c_ListStore->getClassDef()->factory( &ListStore::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_column_types",       &ListStore::set_column_types },
    { "set",                    &ListStore::set },
#if 0 // unused
    { "set_valist",             &ListStore::set_valist },
#endif
    { "set_value",              &ListStore::set_value },
#if 0 // unused
    { "set_valuesv",            &ListStore::set_valuesv },
#endif
    { "remove",                 &ListStore::remove },
    { "insert",                 &ListStore::insert },
    { "insert_before",          &ListStore::insert_before },
    { "insert_after",           &ListStore::insert_after },
    { "insert_with_values",     &ListStore::insert_with_values },
#if 0 // unused
    { "insert_with_valuesv",    &ListStore::insert_with_valuesv },
#endif
    { "prepend",                &ListStore::prepend },
    { "append",                 &ListStore::append },
    { "clear",                  &ListStore::clear },
    { "iter_is_valid",          &ListStore::iter_is_valid },
    { "reorder",                &ListStore::reorder },
    { "swap",                   &ListStore::swap },
    { "move_before",            &ListStore::move_before },
    { "move_after",             &ListStore::move_after },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ListStore, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_ListStore );
    Gtk::TreeModel::clsInit( mod, c_ListStore );
    //Gtk::TreeDragDest::clsInit( mod, c_ListStore );
    //Gtk::TreeDragSource::clsInit( mod, c_ListStore );
    Gtk::TreeSortable::clsInit( mod, c_ListStore );
}


ListStore::ListStore( const Falcon::CoreClass* gen, const GtkListStore* store )
    :
    Gtk::CoreGObject( gen, (GObject*) store )
{}


Falcon::CoreObject* ListStore::factory( const Falcon::CoreClass* gen, void* store, bool )
{
    return new ListStore( gen, (GtkListStore*) store );
}


/*#
    @class GtkListStore
    @brief A list-like data structure that can be used with the GtkTreeView
    @param types an array of GType

    The GtkListStore object is a list model for use with a GtkTreeView widget.
    It implements the GtkTreeModel interface, and consequentialy, can use all of
    the methods available there. It also implements the GtkTreeSortable
    interface so it can be sorted by the view. Finally, it also implements the
    tree drag and drop interfaces.

    The GtkListStore can accept most GObject types as a column type, though it
    can't accept all custom types. Internally, it will keep a copy of data
    passed in (such as a string or a boxed pointer). Columns that accept GObject
    s are handled a little differently. The GtkListStore will keep a reference
    to the object instead of copying the value. As a result, if the object is
    modified, it is up to the application writer to call
    gtk_tree_model_row_changed to emit the "row_changed" signal. This most
    commonly affects lists with GdkPixbufs stored.

    [...]
 */
FALCON_FUNC ListStore::init( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
    CoreArray* arr = i_arr->asArray();
    gint ncol = arr->length();
    GtkListStore* lst;
    if ( ncol == 0 )
        throw_inv_params( "Non-empty array" ); // todo: translate
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
        lst = gtk_list_store_newv( ncol, types );
        memFree( types );
    }
    MYSELF;
    self->setObject( (GObject*) lst );
}


/*#
    @method set_column_types GtkListStore
    @brief Sets the column types.
    @param types an array of GType

    This function is meant primarily for GObjects that inherit from
    GtkListStore, and should only be used when constructing a new GtkListStore.

    It will not function after a row has been added, or a method on the
    GtkTreeModel interface is called.
 */
FALCON_FUNC ListStore::set_column_types( VMARG )
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
        gtk_list_store_set_column_types( (GtkListStore*)_obj, 0, NULL );
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
        gtk_list_store_set_column_types( (GtkListStore*)_obj, ncol, types );
        memFree( types );
    }
}


/*#
    @method set GtkListStore
    @brief Sets the value of one or more cells in the row referenced by iter.
    @param iter row iterator (GtkTreeIter)
    @param values an array of pairs [ column index, column value, ... ]

    The array should contain integer column numbers, each
    column number followed by the value to be set.  For example, to set column 0
    with type G_TYPE_STRING to "Foo", you would write
    gtk_list_store_set ( iter, [ 0, "Foo" ] ).
 */
FALCON_FUNC ListStore::set( VMARG )
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
        gtk_list_store_set( (GtkListStore*)_obj, iter, -1 );
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
                GObject* obj = dyncast<Gtk::CoreGObject*>( it.asObjectSafe() )->getObject();
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
        gtk_list_store_set_valuesv( (GtkListStore*)_obj,
                                    iter, indexes, values, npairs );
        memFree( indexes );
        memFree( values );
    }
}


#if 0 // unused
FALCON_FUNC ListStore::set_valist( VMARG );
#endif


/*#
    @method set_value GtkListStore
    @brief Sets the data in the cell specified by iter and column.
    @param iter A valid GtkTreeIter for the row being modified
    @param column column number to modify
    @param value new value for the cell

    The type of value must be convertible to the type of the column.
 */
FALCON_FUNC ListStore::set_value( VMARG )
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
        GObject* obj = dyncast<Gtk::CoreGObject*>( i_val->asObjectSafe() )->getObject();
        g_value_init( &val, G_TYPE_OBJECT );
        g_value_set_object( &val, obj );
        break;
    }
    default:
        throw_inv_params( "Not implemented" );
    }
    MYSELF;
    GET_OBJ( self );
    gtk_list_store_set_value( (GtkListStore*)_obj,
                              iter, i_col->asInteger(), &val );
}


#if 0 // unused
FALCON_FUNC ListStore::set_valuesv( VMARG );
#endif


/*#
    @method remove GtkListStore
    @brief Removes the given row from the list store.
    @param iter A valid GtkTreeIter
    @return TRUE if iter is valid, FALSE if not.

    After being removed, iter is set to be the next valid row, or invalidated
    if it pointed to the last row in list_store.
 */
FALCON_FUNC ListStore::remove( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_list_store_remove( (GtkListStore*)_obj, iter ) );
}


/*#
    @method insert GtkListStore
    @brief Creates a new row at position.
    @param iter An unset GtkTreeIter to set to the new row
    @param position position to insert the new row

    iter will be changed to point to this new row. If position is larger than
    the number of rows on the list, then the new row will be appended to the
    list. The row will be empty after this function is called. To fill in
    values, you need to call gtk_list_store_set() or gtk_list_store_set_value().
 */
FALCON_FUNC ListStore::insert( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkTreeIter,I" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_list_store_insert( (GtkListStore*)_obj, iter, i_pos->asInteger() );
}


/*#
    @method insert_before GtkListStore
    @brief Inserts a new row before sibling.
    @param iter An unset GtkTreeIter to set to the new row
    @param sibling A valid GtkTreeIter, or NULL.

    If sibling is NULL, then the row will be appended to the end of the list.
    iter will be changed to point to this new row. The row will be empty after
    this function is called. To fill in values, you need to call
    gtk_list_store_set() or gtk_list_store_set_value().
 */
FALCON_FUNC ListStore::insert_before( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_sib = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_sib || !( i_sib->isNil() || ( i_sib->isObject()
        && IS_DERIVED( i_sib, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* sib = i_sib->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_sib->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_list_store_insert_before( (GtkListStore*)_obj, iter, sib );
}


/*#
    @method insert_after GtkListStore
    @brief Inserts a new row after sibling.
    @param iter An unset GtkTreeIter to set to the new row
    @param sibling A valid GtkTreeIter, or NULL.

    If sibling is NULL, then the row will be prepended to the beginning of the
    list. iter will be changed to point to this new row. The row will be empty
    after this function is called. To fill in values, you need to call
    gtk_list_store_set() or gtk_list_store_set_value().
 */
FALCON_FUNC ListStore::insert_after( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_sib = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter )
        || !i_sib || !( i_sib->isNil() || ( i_sib->isObject()
        && IS_DERIVED( i_sib, GtkTreeIter ) ) ) )
        throw_inv_params( "GtkTreeIter,[GtkTreeIter]" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    GtkTreeIter* sib = i_sib->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_sib->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_list_store_insert_after( (GtkListStore*)_obj, iter, sib );
}


/*#
    @method insert_with_values GtkListStore
    @brief Creates a new row at position.
    @param iter An unset GtkTreeIter to set to the new row, or NULL.
    @param pos position to insert the new row
    @param values an array of pairs [ column index, column value, ... ]

    iter will be changed to point to this new row. If position is larger than
    the number of rows on the list, then the new row will be appended to the
    list. The row will be filled with the values given to this function.
 */
FALCON_FUNC ListStore::insert_with_values( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_pos = vm->param( 1 );
    Item* i_pairs = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !( i_iter->isNil() || ( i_iter->isObject()
        && IS_DERIVED( i_iter, GtkTreeIter ) ) )
        || !i_pos || !i_pos->isInteger()
        || !i_pairs || !i_pairs->isArray() )
        throw_inv_params( "[GtkTreeIter],I,A" );
#endif
    GtkTreeIter* iter = i_iter->isNil() ? NULL
                : dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    CoreArray* pairs = i_pairs->asArray();
    const int n = pairs->length();
#ifndef NO_PARAMETER_CHECK
    if ( ( n % 2 ) != 0 )
        throw_inv_params( "pairs of column number and value" ); // todo: translate
#endif
    MYSELF;
    GET_OBJ( self );
    if ( n == 0 )
        gtk_list_store_insert_with_values( (GtkListStore*)_obj,
                                           iter, i_pos->asInteger(), -1 );
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
                GObject* obj = dyncast<Gtk::CoreGObject*>( it.asObjectSafe() )->getObject();
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
        gtk_list_store_insert_with_valuesv( (GtkListStore*)_obj,
                            iter, i_pos->asInteger(), indexes, values, npairs );
        memFree( indexes );
        memFree( values );
    }
}


#if 0 // unused
FALCON_FUNC ListStore::insert_with_valuesv( VMARG );
#endif


/*#
    @method prepend GtkListStore
    @brief Prepends a new row to list_store.
    @param iter An unset GtkTreeIter to set to the prepend row

    iter will be changed to point to this new row. The row will be empty after
    this function is called. To fill in values, you need to call
    gtk_list_store_set() or gtk_list_store_set_value().
 */
FALCON_FUNC ListStore::prepend( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_list_store_prepend( (GtkListStore*)_obj, iter );
}


/*#
    @method append GtkListStore
    @brief Appends a new row to list_store.

    iter will be changed to point to this new row. The row will be empty after
    this function is called. To fill in values, you need to call
    gtk_list_store_set() or gtk_list_store_set_value().
 */
FALCON_FUNC ListStore::append( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    gtk_list_store_append( (GtkListStore*)_obj, iter );
}


/*#
    @method clear GtkListStore
    @brief Removes all rows from the list store.
 */
FALCON_FUNC ListStore::clear( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_list_store_clear( (GtkListStore*)_obj );
}


/*#
    @method iter_is_valid GtkListStore
    @brief Checks if the given iter is a valid iter for this GtkListStore.
    @param iter a GtkTreeIter
    @return TRUE if the iter is valid, FALSE if the iter is invalid.

    Warning: This function is slow. Only use it for debugging and/or testing purposes.
 */
FALCON_FUNC ListStore::iter_is_valid( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || !i_iter->isObject() || !IS_DERIVED( i_iter, GtkTreeIter ) )
        throw_inv_params( "GtkTreeIter" );
#endif
    GtkTreeIter* iter = dyncast<Gtk::TreeIter*>( i_iter->asObjectSafe() )->getTreeIter();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_list_store_iter_is_valid( (GtkListStore*)_obj, iter ) );
}


/*#
    @method reorder GtkListStore
    @brief Reorders store to follow the order indicated by new_order.
    @param new_order an array of integers mapping the new position of each child to its old position before the re-ordering, i.e. new_order[newpos] = oldpos.

    Note that this function only works with unsorted stores.
 */
FALCON_FUNC ListStore::reorder( VMARG )
{
    Item* i_arr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_arr || !i_arr->isArray() )
        throw_inv_params( "A" );
#endif
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
    gtk_list_store_reorder( (GtkListStore*)_obj, order );
    memFree( order );
}


/*#
    @method swap GtkListStore
    @brief Swaps a and b in store. Note that this function only works with unsorted stores.
    @param a A GtkTreeIter.
    @param b Another GtkTreeIter.
 */
FALCON_FUNC ListStore::swap( VMARG )
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
    gtk_list_store_swap( (GtkListStore*)_obj, iter, iter2 );
}


/*#
    @method move_before GtkListStore
    @brief Moves iter in store to the position before position.
    @param iter A GtkTreeIter.
    @param position A GtkTreeIter, or NULL.

    Note that this function only works with unsorted stores. If position is
    NULL, iter will be moved to the end of the list.
 */
FALCON_FUNC ListStore::move_before( VMARG )
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
    gtk_list_store_move_before( (GtkListStore*)_obj, iter, iter2 );
}


/*#
    @method move_after GtkListStore
    @brief Moves iter in store to the position after position.
    @param iter A GtkTreeIter.
    @param position A GtkTreeIter, or NULL.

    Note that this function only works with unsorted stores. If position is
    NULL, iter will be moved to the start of the list.
 */
FALCON_FUNC ListStore::move_after( VMARG )
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
    gtk_list_store_move_after( (GtkListStore*)_obj, iter, iter2 );
}


} // Gtk
} // Falcon
