/**
 *  \file gtk_TextIter.cpp
 */

#include "gtk_TextIter.hpp"

#include "gdk_Pixbuf.hpp"
#include "gtk_TextBuffer.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TextIter::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TextIter = mod->addClass( "GtkTextIter" );

    c_TextIter->setWKS( true );
    c_TextIter->getClassDef()->factory( &TextIter::factory );

    Gtk::MethodTab methods[] =
    {
    { "get_buffer",         &TextIter::get_buffer },
    { "copy",               &TextIter::copy },
    //{ "free",               &TextIter::free },
    { "get_offset",         &TextIter::get_offset },
    { "get_line",           &TextIter::get_line },
    { "get_line_offset",    &TextIter::get_line_offset },
    { "get_line_index",     &TextIter::get_line_index },
    { "get_visible_line_index",&TextIter::get_visible_line_index },
    { "get_visible_line_offset",&TextIter::get_visible_line_offset },
    { "get_char",           &TextIter::get_char },
    { "get_slice",          &TextIter::get_slice },
    { "get_text",           &TextIter::get_text },
    { "get_visible_slice",  &TextIter::get_visible_slice },
    { "get_visible_text",   &TextIter::get_visible_text },
    { "get_pixbuf",         &TextIter::get_pixbuf },
#if 0
    { "get_marks",          &TextIter::get_marks },
    { "get_toggled_tags",        &TextIter:: },
    { "get_child_anchor",        &TextIter:: },
    { "begins_tag",        &TextIter:: },
    { "ends_tag",        &TextIter:: },
    { "toggles_tag",        &TextIter:: },
    { "has_tag",        &TextIter:: },
    { "get_tags",        &TextIter:: },
    { "editable",        &TextIter:: },
    { "can_insert",        &TextIter:: },
    { "starts_word",        &TextIter:: },
    { "ends_word",        &TextIter:: },
    { "inside_word",        &TextIter:: },
    { "starts_line",        &TextIter:: },
    { "ends_line",        &TextIter:: },
    { "starts_sentence",        &TextIter:: },
    { "ends_sentence",        &TextIter:: },
    { "inside_sentence",        &TextIter:: },
    { "is_cursor_position",        &TextIter:: },
    { "get_chars_in_line",        &TextIter:: },
    { "get_bytes_in_line",        &TextIter:: },
    { "get_attributes",        &TextIter:: },
    { "get_language",        &TextIter:: },
    { "is_end",        &TextIter:: },
    { "is_start",        &TextIter:: },
    { "forward_char",        &TextIter:: },
    { "backward_char",        &TextIter:: },
    { "forward_chars",        &TextIter:: },
    { "backward_chars",        &TextIter:: },
    { "forward_line",        &TextIter:: },
    { "backward_line",        &TextIter:: },
    { "forward_lines",        &TextIter:: },
    { "backward_lines",        &TextIter:: },
    { "forward_word_ends",        &TextIter:: },
    { "backward_word_starts",        &TextIter:: },
    { "forward_word_end",        &TextIter:: },
    { "backward_word_start",        &TextIter:: },
    { "forward_cursor_position",        &TextIter:: },
    { "backward_cursor_position",        &TextIter:: },
    { "forward_cursor_positions",        &TextIter:: },
    { "backward_cursor_positions",        &TextIter:: },
    { "backward_sentence_start",        &TextIter:: },
    { "backward_sentence_starts",        &TextIter:: },
    { "forward_sentence_end",        &TextIter:: },
    { "forward_sentence_ends",        &TextIter:: },
    { "forward_visible_word_ends",        &TextIter:: },
    { "backward_visible_word_starts",        &TextIter:: },
    { "forward_visible_word_end",        &TextIter:: },
    { "backward_visible_word_start",        &TextIter:: },
    { "forward_visible_cursor_position",        &TextIter:: },
    { "backward_visible_cursor_position",        &TextIter:: },
    { "forward_visible_cursor_positions",        &TextIter:: },
    { "backward_visible_cursor_positions",        &TextIter:: },
    { "forward_visible_line",        &TextIter:: },
    { "backward_visible_line",        &TextIter:: },
    { "forward_visible_lines",        &TextIter:: },
    { "backward_visible_lines",        &TextIter:: },
    { "set_offset",        &TextIter:: },
    { "set_line",        &TextIter:: },
    { "set_line_offset",        &TextIter:: },
    { "set_line_index",        &TextIter:: },
    { "set_visible_line_index",        &TextIter:: },
    { "set_visible_line_offset",        &TextIter:: },
    { "forward_to_end",        &TextIter:: },
    { "forward_to_line_end",        &TextIter:: },
    { "forward_to_tag_toggle",        &TextIter:: },
    { "backward_to_tag_toggle",        &TextIter:: },
    { "forward_find_char",        &TextIter:: },
    { "backward_find_char",        &TextIter:: },
    { "forward_search",        &TextIter:: },
    { "backward_search",        &TextIter:: },
    { "equal",        &TextIter:: },
    { "compare",        &TextIter:: },
    { "in_range",        &TextIter:: },
    { "order",        &TextIter:: },
#endif
    { NULL, NULL }
    };

    c_TextIter->setWKS( true );
    c_TextIter->getClassDef()->factory( &TextIter::factory );

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextIter, meth->name, meth->cb );
}


TextIter::TextIter( const Falcon::CoreClass* gen, const GtkTextIter* iter )
    :
    Falcon::CoreObject( gen )
{
    if ( iter )
    {
        GtkTextIter* m_iter = (GtkTextIter*) memAlloc( sizeof( GtkTextIter ) );
        *m_iter = *iter;
        setUserData( m_iter );
    }
}


TextIter::~TextIter()
{
    GtkTextIter* iter = (GtkTextIter*) getUserData();
    if ( iter )
        memFree( iter );
}


bool TextIter::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    return defaultProperty( s, it );
}


bool TextIter::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* TextIter::factory( const Falcon::CoreClass* gen, void* iter, bool )
{
    return new TextIter( gen, (GtkTextIter*) iter );
}


/*#
    @class GtkTextIter
    @brief Text buffer iterator
    @optparam table (GtkTextTable) a tag table, or nil to create a new one.

    You may wish to begin by reading the text widget conceptual overview which gives
    an overview of all the objects and data types related to the text widget and how
    they work together.
 */


/*#
    @method get_buffer TextIter
    @brief Returns the GtkTextBuffer this iterator is associated with.
    @return the buffer.
 */
FALCON_FUNC TextIter::get_buffer( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTextBuffer* buf = gtk_text_iter_get_buffer( (GtkTextIter*)_obj );
    vm->retval( new Gtk::TextBuffer( vm->findWKI( "GtkTextBuffer" )->asClass(), buf ) );
}


/*#
    @method copy GtkTextIter
    @brief Creates a copy of an iterator.
    @return a copy of GtkTextIter
 */
FALCON_FUNC TextIter::copy( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTextIter* iter = gtk_text_iter_copy( (GtkTextIter*)_obj );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


//FALCON_FUNC TextIter::free( VMARG );


/*#
    @method get_offset GtkTextIter
    @brief Returns the character offset of an iterator.
    @return a character offset

    Each character in a GtkTextBuffer has an offset, starting with 0 for the first
    character in the buffer. Use gtk_text_buffer_get_iter_at_offset() to convert an
    offset back into an iterator.
 */
FALCON_FUNC TextIter::get_offset( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_iter_get_offset( (GtkTextIter*)_obj ) );
}


/*#
    @method get_line GtkTextIter
    @brief Returns the line number containing the iterator.
    @return a line number

    Lines in a GtkTextBuffer are numbered beginning with 0 for the first line in
    the buffer.
 */
FALCON_FUNC TextIter::get_line( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_iter_get_line( (GtkTextIter*)_obj ) );
}


/*#
    @method get_line_offset GtkTextIter
    @brief Returns the character offset of the iterator, counting from the start of a newline-terminated line.
    @return offset from start of line

    The first character on the line has offset 0.
 */
FALCON_FUNC TextIter::get_line_offset( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_iter_get_line_offset( (GtkTextIter*)_obj ) );
}


/*#
    @method get_line_index
    @brief Returns the byte index of the iterator, counting from the start of a newline-terminated line.
    @return distance from start of line, in bytes

    Remember that GtkTextBuffer encodes text in UTF-8, and that characters can require
    a variable number of bytes to represent.
 */
FALCON_FUNC TextIter::get_line_index( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_iter_get_line_index( (GtkTextIter*)_obj ) );
}


/*#
    @method get_visible_line_index
    @brief Returns the number of bytes from the start of the line to the given iter, not counting bytes that are invisible due to tags with the "invisible" flag toggled on.
    @return byte index of iter with respect to the start of the line
 */
FALCON_FUNC TextIter::get_visible_line_index( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_iter_get_visible_line_index( (GtkTextIter*)_obj ) );
}


/*#
    @method get_visible_line_offset
    @brief Returns the offset in characters from the start of the line to the given iter, not counting characters that are invisible due to tags with the "invisible" flag toggled on.
    @return offset in visible characters from the start of the line
 */
FALCON_FUNC TextIter::get_visible_line_offset( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_iter_get_visible_line_offset( (GtkTextIter*)_obj ) );
}


/*#
    @method get_char
    @brief Returns the Unicode character at this iterator.
    @return a Unicode character, or nil if iter is not dereferenceable
    (Equivalent to operator* on a C++ iterator.) If the element at this iterator is
    a non-character element, such as an image embedded in the buffer, the Unicode
    "unknown" character 0xFFFC is returned. If invoked on the end iterator, zero is
    returned; zero is not a valid Unicode character. So you can write a loop which
    ends when gtk_text_iter_get_char() returns 0.
 */
FALCON_FUNC TextIter::get_char( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gunichar c = gtk_text_iter_get_char( (GtkTextIter*)_obj );
    if ( c )
    {
        String* s = new String( c );
        s->bufferize();
        vm->retval( s );
    }
    else
        vm->retnil();
}


/*#
    @method get_slice
    @brief Returns the text in the given range.
    @param end iterator at end of a range
    @return slice of text from the buffer

    A "slice" is an array of characters encoded in UTF-8 format, including the Unicode
    "unknown" character 0xFFFC for iterable non-character elements in the buffer, such
    as images. Because images are encoded in the slice, byte and character offsets in
    the returned array will correspond to byte offsets in the text buffer.
    Note that 0xFFFC can occur in normal text as well, so it is not a reliable indicator
    that a pixbuf or widget is in the buffer.
 */
FALCON_FUNC TextIter::get_slice( VMARG )
{
    Item* i_end = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_end || i_end->isNil() || !i_end->isObject()
        || !IS_DERIVED( i_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_end )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gchar* ptr = gtk_text_iter_get_slice( (GtkTextIter*)_obj, iter );
    String* s = new String( ptr );
    s->bufferize();
    vm->retval( s );
}


/*#
    @method get_text
    @brief Returns text in the given range.
    @param end iterator at end of a range
    @return array of characters from the buffer

    If the range contains non-text elements such as images, the character and byte
    offsets in the returned string will not correspond to character and byte offsets
    in the buffer. If you want offsets to correspond, see gtk_text_iter_get_slice().
 */
FALCON_FUNC TextIter::get_text( VMARG )
{
    Item* i_end = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_end || i_end->isNil() || !i_end->isObject()
        || !IS_DERIVED( i_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_end )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gchar* ptr = gtk_text_iter_get_text( (GtkTextIter*)_obj, iter );
    String* s = new String( ptr );
    s->bufferize();
    vm->retval( s );
}


/*#
    @method get_visible_slice
    @brief Like get_slice(), but invisible text is not included.
    @param end iterator at end of a range
    @return slice of text from the buffer

    Invisible text is usually invisible because a GtkTextTag with the "invisible"
    attribute turned on has been applied to it.
 */
FALCON_FUNC TextIter::get_visible_slice( VMARG )
{
    Item* i_end = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_end || i_end->isNil() || !i_end->isObject()
        || !IS_DERIVED( i_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_end )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gchar* ptr = gtk_text_iter_get_visible_slice( (GtkTextIter*)_obj, iter );
    String* s = new String( ptr );
    s->bufferize();
    vm->retval( s );
}


/*#
    @method Like gtk_text_iter_get_text(), but invisible text is not included.
    @param end iterator at end of a range
    @return string containing visible text in the range

    Invisible text is usually invisible because a GtkTextTag with the "invisible"
    attribute turned on has been applied to it.
 */
FALCON_FUNC TextIter::get_visible_text( VMARG )
{
    Item* i_end = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_end || i_end->isNil() || !i_end->isObject()
        || !IS_DERIVED( i_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_end )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gchar* ptr = gtk_text_iter_get_visible_text( (GtkTextIter*)_obj, iter );
    String* s = new String( ptr );
    s->bufferize();
    vm->retval( s );
}


/*#
    @method get_pixbuf
    @brief If the element at iter is a pixbuf, the pixbuf is returned. Otherwise, nil is returned.
    @return the pixbuf
 */
FALCON_FUNC TextIter::get_pixbuf( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GdkPixbuf* buf = gtk_text_iter_get_pixbuf( (GtkTextIter*)_obj );
    if ( buf )
        vm->retval( new Gdk::Pixbuf( vm->findWKI( "GdkPixBuf" )->asClass(), buf ) );
    else
        vm->retnil();
}


#if 0
FALCON_FUNC TextIter::get_marks( VMARG );

FALCON_FUNC TextIter::get_toggled_tags( VMARG );

FALCON_FUNC TextIter::get_child_anchor( VMARG );

FALCON_FUNC TextIter::begins_tag( VMARG );

FALCON_FUNC TextIter::ends_tag( VMARG );

FALCON_FUNC TextIter::toggles_tag( VMARG );

FALCON_FUNC TextIter::has_tag( VMARG );

FALCON_FUNC TextIter::get_tags( VMARG );

FALCON_FUNC TextIter::editable( VMARG );

FALCON_FUNC TextIter::can_insert( VMARG );

FALCON_FUNC TextIter::starts_word( VMARG );

FALCON_FUNC TextIter::ends_word( VMARG );

FALCON_FUNC TextIter::inside_word( VMARG );

FALCON_FUNC TextIter::starts_line( VMARG );

FALCON_FUNC TextIter::ends_line( VMARG );

FALCON_FUNC TextIter::starts_sentence( VMARG );

FALCON_FUNC TextIter::ends_sentence( VMARG );

FALCON_FUNC TextIter::inside_sentence( VMARG );

FALCON_FUNC TextIter::is_cursor_position( VMARG );

FALCON_FUNC TextIter::get_chars_in_line( VMARG );

FALCON_FUNC TextIter::get_bytes_in_line( VMARG );

FALCON_FUNC TextIter::get_attributes( VMARG );

FALCON_FUNC TextIter::get_language( VMARG );

FALCON_FUNC TextIter::is_end( VMARG );

FALCON_FUNC TextIter::is_start( VMARG );

FALCON_FUNC TextIter::forward_char( VMARG );

FALCON_FUNC TextIter::backward_char( VMARG );

FALCON_FUNC TextIter::forward_chars( VMARG );

FALCON_FUNC TextIter::backward_chars( VMARG );

FALCON_FUNC TextIter::forward_line( VMARG );

FALCON_FUNC TextIter::backward_line( VMARG );

FALCON_FUNC TextIter::forward_lines( VMARG );

FALCON_FUNC TextIter::backward_lines( VMARG );

FALCON_FUNC TextIter::forward_word_ends( VMARG );

FALCON_FUNC TextIter::backward_word_starts( VMARG );

FALCON_FUNC TextIter::forward_word_end( VMARG );

FALCON_FUNC TextIter::backward_word_start( VMARG );

FALCON_FUNC TextIter::forward_cursor_position( VMARG );

FALCON_FUNC TextIter::backward_cursor_position( VMARG );

FALCON_FUNC TextIter::forward_cursor_positions( VMARG );

FALCON_FUNC TextIter::backward_cursor_positions( VMARG );

FALCON_FUNC TextIter::backward_sentence_start( VMARG );

FALCON_FUNC TextIter::backward_sentence_starts( VMARG );

FALCON_FUNC TextIter::forward_sentence_end( VMARG );

FALCON_FUNC TextIter::forward_sentence_ends( VMARG );

FALCON_FUNC TextIter::forward_visible_word_ends( VMARG );

FALCON_FUNC TextIter::backward_visible_word_starts( VMARG );

FALCON_FUNC TextIter::forward_visible_word_end( VMARG );

FALCON_FUNC TextIter::backward_visible_word_start( VMARG );

FALCON_FUNC TextIter::forward_visible_cursor_position( VMARG );

FALCON_FUNC TextIter::backward_visible_cursor_position( VMARG );

FALCON_FUNC TextIter::forward_visible_cursor_positions( VMARG );

FALCON_FUNC TextIter::backward_visible_cursor_positions( VMARG );

FALCON_FUNC TextIter::forward_visible_line( VMARG );

FALCON_FUNC TextIter::backward_visible_line( VMARG );

FALCON_FUNC TextIter::forward_visible_lines( VMARG );

FALCON_FUNC TextIter::backward_visible_lines( VMARG );

FALCON_FUNC TextIter::set_offset( VMARG );

FALCON_FUNC TextIter::set_line( VMARG );

FALCON_FUNC TextIter::set_line_offset( VMARG );

FALCON_FUNC TextIter::set_line_index( VMARG );

FALCON_FUNC TextIter::set_visible_line_index( VMARG );

FALCON_FUNC TextIter::set_visible_line_offset( VMARG );

FALCON_FUNC TextIter::forward_to_end( VMARG );

FALCON_FUNC TextIter::forward_to_line_end( VMARG );

FALCON_FUNC TextIter::forward_to_tag_toggle( VMARG );

FALCON_FUNC TextIter::backward_to_tag_toggle( VMARG );

FALCON_FUNC TextIter::forward_find_char( VMARG );

FALCON_FUNC TextIter::backward_find_char( VMARG );

FALCON_FUNC TextIter::forward_search( VMARG );

FALCON_FUNC TextIter::backward_search( VMARG );

FALCON_FUNC TextIter::equal( VMARG );

FALCON_FUNC TextIter::compare( VMARG );

FALCON_FUNC TextIter::in_range( VMARG );

FALCON_FUNC TextIter::order( VMARG );

#endif

} // Gtk
} // Falcon
