/**
 *  \file gtk_TextMark.cpp
 */

#include "gtk_TextMark.hpp"

#include "gtk_TextBuffer.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TextMark::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TextMark = mod->addClass( "GtkTextMark", &TextMark::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_TextMark->getClassDef()->addInheritance( in );

    c_TextMark->setWKS( true );
    c_TextMark->getClassDef()->factory( &TextMark::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_visible",        &TextMark::set_visible },
    { "get_visible",        &TextMark::get_visible },
    { "get_deleted",        &TextMark::get_deleted },
    { "get_name",           &TextMark::get_name },
    { "get_buffer",         &TextMark::get_buffer },
    { "get_left_gravity",   &TextMark::get_left_gravity },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextMark, meth->name, meth->cb );
}


TextMark::TextMark( const Falcon::CoreClass* gen, const GtkTextMark* mk )
    :
    Gtk::CoreGObject( gen, (GObject*) mk )
{}


Falcon::CoreObject* TextMark::factory( const Falcon::CoreClass* gen, void* mk, bool )
{
    return new TextMark( gen, (GtkTextMark*) mk );
}


/*#
    @class GtkTextMark
    @brief A position in the buffer preserved across buffer modifications
    @optparam name mark name or nil.
    @optparam left_gravity (boolean) whether the mark should have left gravity (default false)

    You may wish to begin by reading the text widget conceptual overview which gives
    an overview of all the objects and data types related to the text widget and how
    they work together.

    A GtkTextMark is like a bookmark in a text buffer; it preserves a position in
    the text. You can convert the mark to an iterator using gtk_text_buffer_get_iter_at_mark().
    Unlike iterators, marks remain valid across buffer mutations, because their behavior
    is defined when text is inserted or deleted. When text containing a mark is deleted,
    the mark remains in the position originally occupied by the deleted text.
    When text is inserted at a mark, a mark with left gravity will be moved to the
    beginning of the newly-inserted text, and a mark with right gravity will be moved
    to the end.

    Marks are reference counted, but the reference count only controls the validity
    of the memory; marks can be deleted from the buffer at any time with
    gtk_text_buffer_delete_mark(). Once deleted from the buffer, a mark is
    essentially useless.

    Marks optionally have names; these can be convenient to avoid passing the
    GtkTextMark object around.

    Marks are typically created using the gtk_text_buffer_create_mark() function.

    @note Creating a text mark: Add it to a buffer using gtk_text_buffer_add_mark().
    If name is nil, the mark is anonymous; otherwise, the mark can be retrieved by name
    using gtk_text_buffer_get_mark(). If a mark has left gravity, and text is inserted
    at the mark's current location, the mark will be moved to the left of the
    newly-inserted text. If the mark has right gravity (left_gravity = false),
    the mark will end up on the right of newly-inserted text. The standard
    left-to-right cursor is a mark with right gravity (when you type, the cursor
    stays on the right side of the text you're typing).

 */
FALCON_FUNC TextMark::init( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S,B]" );

    char* name = args.getCString( 0, false );
    gboolean gravity = args.getBoolean( 1, false );

    MYSELF;
    GtkTextMark* mk = gtk_text_mark_new( name, gravity );
    self->setObject( (GObject*) mk );
}


/*#
    @method set_visible GtkTextMark
    @brief Sets the visibility of mark.
    @param setting visibility of mark
    The insertion point is normally visible, i.e. you can see it as a vertical bar.
    Also, the text widget uses a visible mark to indicate where a drop will occur
    when dragging-and-dropping text. Most other marks are not visible. Marks are
    not visible by default.
 */
FALCON_FUNC TextMark::set_visible( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_mark_set_visible( (GtkTextMark*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_visible GtkTextMark
    @brief Returns true if the mark is visible (i.e. a cursor is displayed for it).
    @return true if visible
 */
FALCON_FUNC TextMark::get_visible( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_mark_get_visible( (GtkTextMark*)_obj ) );
}


/*#
    @method get_deleted GtkTextMark
    @brief Returns true if the mark has been removed from its buffer with gtk_text_buffer_delete_mark().
    @return whether the mark is deleted

    See gtk_text_buffer_add_mark() for a way to add it to a buffer again.
 */
FALCON_FUNC TextMark::get_deleted( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_mark_get_deleted( (GtkTextMark*)_obj ) );
}


/*#
    @method get_name GtkTextMark
    @brief Returns the mark name; returns nil for anonymous marks.
    @return mark name
 */
FALCON_FUNC TextMark::get_name( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const char* name = gtk_text_mark_get_name( (GtkTextMark*)_obj );
    if ( name )
    {
        String* s = new String( name );
        s->bufferize();
        vm->retval( s );
    }
    else
        vm->retnil();
}


/*#
    @method get_buffer GtkTextMark
    @brief Gets the buffer this mark is located inside, or nil if the mark is deleted.
    @return the mark's GtkTextBuffer.
 */
FALCON_FUNC TextMark::get_buffer( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkTextBuffer* buf = gtk_text_mark_get_buffer( (GtkTextMark*)_obj );
    if ( buf )
        vm->retval( new Gtk::TextBuffer( vm->findWKI( "GtkTextBuffer" )->asClass(), buf ) );
    else
        vm->retnil();
}


/*#
    @method get_left_gravity GtkTextMark
    @brief Determines whether the mark has left gravity.
    @return true if the mark has left gravity, false otherwise
 */
FALCON_FUNC TextMark::get_left_gravity( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_mark_get_left_gravity( (GtkTextMark*)_obj ) );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
