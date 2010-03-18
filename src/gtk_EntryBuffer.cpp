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

    Gtk::MethodTab methods[] =
    {
    { "get_text",           &EntryBuffer::get_text },
    { "set_text",           &EntryBuffer::set_text },
    { "get_bytes",          &EntryBuffer::get_bytes },
    { "get_length",         &EntryBuffer::get_length },
    { "get_max_length",     &EntryBuffer::get_max_length },
    { "set_max_length",     &EntryBuffer::set_max_length },
    //{ "insert_text",        &EntryBuffer::insert_text },
    //{ "delete_text",        &EntryBuffer::delete_text },
    //{ "emit_deleted_text",  &EntryBuffer::emit_deleted_text },
    //{ "emit_inserted_text", &EntryBuffer::emit_inserted_text },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_EntryBuffer, meth->name, meth->cb );
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


/*#
    @method get_text gtk.EntryBuffer
    @brief Retrieves the contents of the buffer.
    @return (string) contents of buffer

    The memory pointer returned by this call will not change unless this object
    emits a signal, or is finalized.
 */
FALCON_FUNC EntryBuffer::get_text( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* txt = gtk_entry_buffer_get_text( (GtkEntryBuffer*)_obj );
    vm->retval( new String( txt ) );
}


/*#
    @method set_text gtk.EntryBuffer
    @brief Sets the text in the buffer.
    @param text the new text

    This is roughly equivalent to calling delete_text() and insert_text().
 */
FALCON_FUNC EntryBuffer::set_text( VMARG )
{
    Item* i_txt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString s( i_txt->asString() );
    gtk_entry_buffer_set_text( (GtkEntryBuffer*)_obj, s.c_str(), strlen( s.c_str() ) );
}


/*#
    @method get_bytes gtk.EntryBuffer
    @brief Retrieves the length in bytes of the buffer.
    @return The byte length of the buffer.

    See get_length().
 */
FALCON_FUNC EntryBuffer::get_bytes( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_entry_buffer_get_bytes( (GtkEntryBuffer*)_obj ) );
}


/*#
    @method get_length gtk.EntryBuffer
    @brief Retrieves the length in characters of the buffer.
    @return The number of characters in the buffer.
 */
FALCON_FUNC EntryBuffer::get_length( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_entry_buffer_get_length( (GtkEntryBuffer*)_obj ) );
}


/*#
    @method get_max_length gtk.EntryBuffer
    @brief Retrieves the maximum allowed length of the text in buffer.
    @return the maximum allowed number of characters in GtkEntryBuffer, or 0 if there is no maximum.

    See set_max_length().
 */
FALCON_FUNC EntryBuffer::get_max_length( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_entry_buffer_get_max_length( (GtkEntryBuffer*)_obj ) );
}


/*#
    @method set_max_length gtk.EntryBuffer
    @brief Sets the maximum allowed length of the contents of the buffer.
    @param max_length the maximum length of the entry buffer, or 0 for no maximum.

    If the current contents are longer than the given length, then they will be
    truncated to fit.
 */
FALCON_FUNC EntryBuffer::set_max_length( VMARG )
{
    Item* i_len = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_len || i_len->isNil() || !i_len->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_buffer_set_max_length( (GtkEntryBuffer*)_obj, i_len->asInteger() );
}


//FALCON_FUNC EntryBuffer::insert_text( VMARG );

//FALCON_FUNC EntryBuffer::delete_text( VMARG );

//FALCON_FUNC EntryBuffer::emit_deleted_text( VMARG );

//FALCON_FUNC EntryBuffer::emit_inserted_text( VMARG );


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 18
