/**
 *  \file gtk_TextTag.cpp
 */

#include "gtk_TextTag.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TextTag::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TextTag = mod->addClass( "GtkTextTag", &TextTag::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_TextTag->getClassDef()->addInheritance( in );

    c_TextTag->setWKS( true );
    c_TextTag->getClassDef()->factory( &TextTag::factory );

    Gtk::MethodTab methods[] =
    {
    //{ "signal_event",   &TextTag::signal_event },
    { "get_priority",   &TextTag::get_priority },
    { "set_priority",   &TextTag::set_priority },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextTag, meth->name, meth->cb );
}


TextTag::TextTag( const Falcon::CoreClass* gen, const GtkTextTag* tag )
    :
    Gtk::CoreGObject( gen, (GObject*) tag )
{}


Falcon::CoreObject* TextTag::factory( const Falcon::CoreClass* gen, void* tag, bool )
{
    return new TextTag( gen, (GtkTextTag*) tag );
}


/*#
    @class GtkTextTag
    @brief A tag that can be applied to text in a GtkTextBuffer
    @optparam name tag name

    You may wish to begin by reading the text widget conceptual overview which gives
    an overview of all the objects and data types related to the text widget and how
    they work together.

    Tags should be in the GtkTextTagTable for a given GtkTextBuffer before using them
    with that buffer.

    gtk_text_buffer_create_tag() is the best way to create tags. See gtk-demo for
    numerous examples.
 */
FALCON_FUNC TextTag::init( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );

    char* name = args.getCString( 0, false );

    MYSELF;
    GtkTextTag* tag = gtk_text_tag_new( name );
    self->setGObject( (GObject*) tag );
}


//FALCON_FUNC TextTag::signal_event( VMARG );


/*#
    @method get_priority GtkTextTag
    @brief Get the tag priority.
    @return The tag's priority.
 */
FALCON_FUNC TextTag::get_priority( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_tag_get_priority( (GtkTextTag*)_obj ) );
}


/*#
    @method set_priority GtkTextTag
    @brief Sets the priority of a GtkTextTag.
    @param priority the new priority

    Valid priorities are start at 0 and go to one less than gtk_text_tag_table_get_size().
    Each tag in a table has a unique priority; setting the priority of one tag shifts
    the priorities of all the other tags in the table to maintain a unique priority
    for each tag. Higher priority tags "win" if two tags both set the same text attribute.
    When adding a tag to a tag table, it will be assigned the highest priority in the table
    by default; so normally the precedence of a set of tags is the order in which they were
    added to the table, or created with gtk_text_buffer_create_tag(), which adds the tag
    to the buffer's table automatically.
 */
FALCON_FUNC TextTag::set_priority( VMARG )
{
    Item* i_prio = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_prio || i_prio->isNil() || !i_prio->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_tag_set_priority( (GtkTextTag*)_obj, i_prio->asInteger() );
}


} // Gtk
} // Falcon
