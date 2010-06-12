/**
 *  \file gtk_ListStore.cpp
 */

#include "gtk_ListStore.hpp"

#include "gtk_Buildable.hpp"
//#include "gtk_TreeModel.hpp"
//#include "gtk_TreeDragDest.hpp"
//#include "gtk_TreeDragSource.hpp"
//#include "gtk_TreeSortable.hpp"


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

    c_ListStore->setWKS( true );
    c_ListStore->getClassDef()->factory( &ListStore::factory );

    Gtk::MethodTab methods[] =
    {
#if 0
    { "set_column_types",    &ListStore:: },
    { "set",    &ListStore:: },
    { "set_valist",    &ListStore:: },
    { "set_value",    &ListStore:: },
    { "set_valuesv",    &ListStore:: },
    { "remove",    &ListStore:: },
    { "insert",    &ListStore:: },
    { "insert_before",    &ListStore:: },
    { "insert_after",    &ListStore:: },
    { "insert_with_values",    &ListStore:: },
    { "insert_with_valuesv",    &ListStore:: },
    { "prepend",    &ListStore:: },
    { "append",    &ListStore:: },
    { "clear",    &ListStore:: },
    { "iter_is_valid",    &ListStore:: },
    { "reorder",    &ListStore:: },
    { "swap",    &ListStore:: },
    { "move_before",    &ListStore:: },
    { "move_after",    &ListStore:: },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ListStore, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_ListStore );
    //Gtk::TreeModel::clsInit( mod, c_Liststore );
    //Gtk::TreeDragDest::clsInit( mod, c_ListStore );
    //Gtk::TreeDragSource::clsInit( mod, c_ListStore );
    //Gtk::TreeSortable::clsInit( mod, c_ListStore );
}


ListStore::ListStore( const Falcon::CoreClass* gen, const GtkListStore* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* ListStore::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new ListStore( gen, (GtkListStore*) btn );
}


/*#
    @class GtkListStore
    @brief A list-like data structure that can be used with the GtkTreeView
    @param an array of GType

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
        lst = gtk_list_store_newv( 0, NULL );
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
    self->setGObject( (GObject*) lst );
}


#if 0
FALCON_FUNC ListStore::set_column_types( VMARG );
FALCON_FUNC ListStore::set( VMARG );
FALCON_FUNC ListStore::set_valist( VMARG );
FALCON_FUNC ListStore::set_value( VMARG );
FALCON_FUNC ListStore::set_valuesv( VMARG );
FALCON_FUNC ListStore::remove( VMARG );
FALCON_FUNC ListStore::insert( VMARG );
FALCON_FUNC ListStore::insert_before( VMARG );
FALCON_FUNC ListStore::insert_after( VMARG );
FALCON_FUNC ListStore::insert_with_values( VMARG );
FALCON_FUNC ListStore::insert_with_valuesv( VMARG );
FALCON_FUNC ListStore::prepend( VMARG );
FALCON_FUNC ListStore::append( VMARG );
FALCON_FUNC ListStore::clear( VMARG );
FALCON_FUNC ListStore::iter_is_valid( VMARG );
FALCON_FUNC ListStore::reorder( VMARG );
FALCON_FUNC ListStore::swap( VMARG );
FALCON_FUNC ListStore::move_before( VMARG );
FALCON_FUNC ListStore::move_after( VMARG );
#endif

} // Gtk
} // Falcon
