/**
 *  \file gtk_EntryBuffer.cpp
 */

#include "gtk_EntryBuffer.hpp"

#include <gtk/gtk.h>

#if GTK_MINOR_VERSION >= 18

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void EntryBuffer::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_EntryBuffer = mod->addClass( "EntryBuffer", &EntryBuffer::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_EntryBuffer->getClassDef()->addInheritance( in );

    c_EntryBuffer->setWKS( true );
    c_EntryBuffer->getClassDef()->factory( &EntryBuffer::factory );
#if 0
    Gtk::MethodTab methods[] =
    {
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_EntryBuffer, meth->name, meth->cb );
#endif
}


EntryBuffer::EntryBuffer( const Falcon::CoreClass* gen, const GtkEntryBuffer* buf )
    :
    Gtk::CoreGObject( gen )
{
    if ( buf )
        setUserData( new GData( (GObject*) buf ) );
}


Falcon::CoreObject* EntryBuffer::factory( const Falcon::CoreClass* gen, void* buf, bool )
{
    return new EntryBuffer( gen, (GtkEntryBuffer*) buf );
}


/*#
    @class gtk.EntryBuffer
    @brief Text buffer for GtkEntry

    The gtk.EntryBuffer class contains the actual text displayed in a gtk.Entry widget.

    A single gtk.EntryBuffer object can be shared by multiple gtk.Entry widgets which
    will then share the same text content, but not the cursor position, visibility
    attributes, icon etc.

    gtk.EntryBuffer may be derived from. Such a derived class might allow text to be
    stored in an alternate location, such as non-pageable memory, useful in the case
    of important passwords. Or a derived class could integrate with an application's
    concept of undo/redo.
 */

/*#
    @init gtk.EntryBuffer
    @brief Create a new gtk.EntryBuffer object.
    @optparam initial_text (string) Optionally, specify initial text to set in the buffer.
 */
FALCON_FUNC EntryBuffer::init( VMARG )
{
    MYSELF;
    Item* i_txt = vm->param( 0 );
    GtkEntryBuffer* buf;
    if ( i_txt )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_txt->isNil() || !i_txt->isString() )
            throw_inv_params( "[S]" );
#endif
        AutoCString s( i_txt->asString() );
        buf = gtk_entry_buffer_new( s.c_str(), strlen( s.c_str() ) );
    }
    else
        buf = gtk_entry_buffer_new( NULL, -1 );

    Gtk::internal_add_slot( (GObject*) buf );
    self->setUserData( new GData( (GObject*) buf ) );
}


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 18
