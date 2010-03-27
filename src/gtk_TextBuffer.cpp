/**
 *  \file gtk_TextBuffer.cpp
 */

#include "gtk_TextBuffer.hpp"

#include "gtk_TextTagTable.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TextBuffer::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TextBuffer = mod->addClass( "GtkTextBuffer", &TextBuffer::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_TextBuffer->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    //
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextBuffer, meth->name, meth->cb );
}


TextBuffer::TextBuffer( const Falcon::CoreClass* gen, const GtkTextBuffer* buf )
    :
    Gtk::CoreGObject( gen )
{
    if ( buf )
        setUserData( new GData( (GObject*) buf ) );
}


Falcon::CoreObject* TextBuffer::factory( const Falcon::CoreClass* gen, void* buf, bool )
{
    return new TextBuffer( gen, (GtkTextBuffer*) buf );
}


/*#
    @class GtkTextBuffer
    @brief Stores attributed text for display in a GtkTextView
    @optparam table (GtkTextTable) a tag table, or nil to create a new one.

    You may wish to begin by reading the text widget conceptual overview which gives
    an overview of all the objects and data types related to the text widget and how
    they work together.
 */
FALCON_FUNC TextBuffer::init( VMARG )
{
    Item* i_tab = vm->param( 0 );
    // this method accepts nil
    GtkTextTagTable* tab = NULL;

    if ( i_tab )
    {
        if ( !i_tab->isNil() )
        {
#ifndef NO_PARAMETER_CHECK
            if ( !i_tab->isObject() || !IS_DERIVED( i_tab, GtkTextTagTable ) )
                throw_inv_params( "[GtkTextTagTable]" );
#endif
            tab = (GtkTextTagTable*)((GData*)i_tab->asObject()->getUserData())->obj();
        }
    }

    MYSELF;
    GtkTextBuffer* buf = gtk_text_buffer_new( tab );
    Gtk::internal_add_slot( (GObject*) buf );
    self->setUserData( new GData( (GObject*) buf ) );
}


} // Gtk
} // Falcon
