/**
 *  \file gtk_TextTagTable.cpp
 */

#include "gtk_TextTagTable.hpp"

#include "gtk_TextTag.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TextTagTable::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TextTagTable = mod->addClass( "GtkTextTagTable", &TextTagTable::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_TextTagTable->getClassDef()->addInheritance( in );

    c_TextTagTable->setWKS( true );
    c_TextTagTable->getClassDef()->factory( &TextTagTable::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_tag_added",   &TextTagTable::signal_tag_added },
    //{ "signal_tag_changed", &TextTagTable::signal_tag_changed },
    { "signal_tag_removed", &TextTagTable::signal_tag_removed },
    { "add",                &TextTagTable::add },
    { "remove",             &TextTagTable::remove },
    { "lookup",             &TextTagTable::lookup },
    //{ "foreach",           &TextTagTable::foreach },
    { "get_size",           &TextTagTable::get_size },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextTagTable, meth->name, meth->cb );
}


TextTagTable::TextTagTable( const Falcon::CoreClass* gen, const GtkTextTagTable* tag )
    :
    Gtk::CoreGObject( gen, (GObject*) tag )
{}


Falcon::CoreObject* TextTagTable::factory( const Falcon::CoreClass* gen, void* tag, bool )
{
    return new TextTagTable( gen, (GtkTextTagTable*) tag );
}


/*#
    @class GtkTextTagTable
    @brief Collection of tags that can be used together

    You may wish to begin by reading the text widget conceptual overview which gives
    an overview of all the objects and data types related to the text widget and
    how they work together.
 */
FALCON_FUNC TextTagTable::init( VMARG )
{
    NO_ARGS
    MYSELF;
    GtkTextTagTable* tab = gtk_text_tag_table_new();
    self->setGObject( (GObject*) tab );
}


/*#
    @method signal_tag_added GtkTextTagTable
    @brief Connect a VMSlot to the table tag-added signal and return it
 */
FALCON_FUNC TextTagTable::signal_tag_added( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "tag_added", (void*) &TextTagTable::on_tag_added, vm );
}


void TextTagTable::on_tag_added( GtkTextTagTable* obj, GtkTextTag* tag, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "tag_added", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkTextTag" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_tag_added", it ) )
            {
                printf(
                "[GtkTextTagTable::on_tag_added] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TextTag( wki->asClass(), tag ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}

#if 0
/*#
    @method signal_tag_changed GtkTextTagTable
    @brief Connect a VMSlot to the table tag-changed signal and return it
 */
FALCON_FUNC TextTagTable::signal_tag_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "tag_changed", (void*) &TextTagTable::on_tag_changed, vm );
}


void TextTagTable::on_tag_changed( GtkTextTagTable* obj, GtkTextTag* tag, gboolean arg2, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "tag_changed", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_tag_changed", it ) )
            {
                printf(
                "[GtkTextTagTable::on_tag_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (bool) arg2 );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}
#endif

/*#
    @method signal_tag_removed GtkTextTagTable
    @brief Connect a VMSlot to the table tag-removed signal and return it
 */
FALCON_FUNC TextTagTable::signal_tag_removed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "tag_removed", (void*) &TextTagTable::on_tag_removed, vm );
}


void TextTagTable::on_tag_removed( GtkTextTagTable* obj, GtkTextTag* tag, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "tag_removed", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkTextTag" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_tag_removed", it ) )
            {
                printf(
                "[GtkTextTagTable::on_tag_removed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::TextTag( wki->asClass(), tag ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method add GtkTextTagTable
    @brief Add a tag to the table. The tag is assigned the highest priority in the table.
    @param tag a GtkTextTag

    tag must not be in a tag table already, and may not have the same name as an
    already-added tag.
 */
FALCON_FUNC TextTagTable::add( VMARG )
{
    Item* i_tag = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tag || i_tag->isNil() || !i_tag->isObject()
        || !IS_DERIVED( i_tag, GtkTextTag ) )
        throw_inv_params( "GtkTextTag" );
#endif
    GtkTextTag* tag = (GtkTextTag*) COREGOBJECT( i_tag )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_tag_table_add( (GtkTextTagTable*)_obj, tag );
}


/*#
    @method remove GtkTextTagTable
    @brief Remove a tag from the table.
    @param tag a GtkTextTag
 */
FALCON_FUNC TextTagTable::remove( VMARG )
{
    Item* i_tag = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tag || i_tag->isNil() || !i_tag->isObject()
        || !IS_DERIVED( i_tag, GtkTextTag ) )
        throw_inv_params( "GtkTextTag" );
#endif
    GtkTextTag* tag = (GtkTextTag*) COREGOBJECT( i_tag )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_tag_table_remove( (GtkTextTagTable*)_obj, tag );
}

/*#
    @method lookup GtkTextTagTable
    @brief Look up a named tag.
    @param name name of a tag
    @return The tag, or nil if none by that name is in the table
 */
FALCON_FUNC TextTagTable::lookup( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );

    char* name = args.getCString( 0 );

    MYSELF;
    GET_OBJ( self );
    GtkTextTag* tag = gtk_text_tag_table_lookup( (GtkTextTagTable*)_obj, name );
    if ( tag )
        vm->retval( new Gtk::TextTag( vm->findWKI( "GtkTextTag" )->asClass(), tag ) );
    else
        vm->retnil();
}


//FALCON_FUNC TextTagTable::foreach( VMARG );


/*#
    @method get_size GtkTextTagTable
    @brief Returns the size of the table (number of tags)
    @return number of tags in table
 */
FALCON_FUNC TextTagTable::get_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_tag_table_get_size( (GtkTextTagTable*)_obj ) );
}


} // Gtk
} // Falcon
