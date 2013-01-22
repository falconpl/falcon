/**
 *  \file gtk_TextBuffer.cpp
 */

#include "gtk_TextBuffer.hpp"

#include "gtk_TextIter.hpp"
#include "gtk_TextMark.hpp"
#include "gtk_TextTagTable.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

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

    c_TextBuffer->setWKS( true );
    c_TextBuffer->getClassDef()->factory( &TextBuffer::factory );

    Gtk::MethodTab methods[] =
    {
    { "get_line_count",         &TextBuffer::get_line_count },
    { "get_char_count",         &TextBuffer::get_char_count },
    { "get_tag_table",          &TextBuffer::get_tag_table },
    { "insert",                 &TextBuffer::insert },
    { "insert_at_cursor",       &TextBuffer::insert_at_cursor },
    { "insert_interactive",     &TextBuffer::insert_interactive },
    { "insert_interactive_at_cursor",&TextBuffer::insert_interactive_at_cursor },
    { "insert_range",           &TextBuffer::insert_range },
    { "insert_range_interactive",&TextBuffer::insert_range_interactive },
    //{ "insert_with_tags",       &TextBuffer::insert_with_tags },
    //{ "insert_with_tags_by_name",&TextBuffer::insert_with_tags_by_name },
    { "delete_",                &TextBuffer::delete_ },
    { "delete_interactive",     &TextBuffer::delete_interactive },
    { "backspace",              &TextBuffer::backspace },
    { "set_text",               &TextBuffer::set_text },
    { "get_text",               &TextBuffer::get_text },
    { "get_slice",              &TextBuffer::get_slice },
    { "insert_pixbuf",          &TextBuffer::insert_pixbuf },
    //{ "insert_child_anchor",    &TextBuffer::insert_child_anchor },
    //{ "create_child_anchor",    &TextBuffer::create_child_anchor },
    { "create_mark",            &TextBuffer::create_mark },
    { "move_mark",              &TextBuffer::move_mark },
    { "move_mark_by_name",      &TextBuffer::move_mark_by_name },
    { "add_mark",               &TextBuffer::add_mark },
    { "delete_mark",            &TextBuffer::delete_mark },
    { "delete_mark_by_name",    &TextBuffer::delete_mark_by_name },
    { "get_mark",               &TextBuffer::get_mark },
    { "get_insert",             &TextBuffer::get_insert },
    { "get_selection_bound",    &TextBuffer::get_selection_bound },
    { "get_has_selection",      &TextBuffer::get_has_selection },
    { "place_cursor",           &TextBuffer::place_cursor },
    { "select_range",           &TextBuffer::select_range },
    { "apply_tag",              &TextBuffer::apply_tag },
    { "remove_tag",             &TextBuffer::remove_tag },
    { "apply_tag_by_name",      &TextBuffer::apply_tag_by_name },
    { "remove_tag_by_name",     &TextBuffer::remove_tag_by_name },
    { "remove_all_tags",        &TextBuffer::remove_all_tags },
    //{ "create_tag",             &TextBuffer::create_tag },
    { "get_iter_at_line_offset",&TextBuffer::get_iter_at_line_offset },
    { "get_iter_at_offset",     &TextBuffer::get_iter_at_offset },
    { "get_iter_at_line",       &TextBuffer::get_iter_at_line },
    { "get_iter_at_line_index", &TextBuffer::get_iter_at_line_index },
    //{ "get_iter_at_mark",       &TextBuffer::get_iter_at_mark },
    //{ "get_iter_at_child_anchor",&TextBuffer::get_iter_at_child_anchor },
    { "get_start_iter",         &TextBuffer::get_start_iter },
    { "get_end_iter",           &TextBuffer::get_end_iter },
    { "get_bounds",             &TextBuffer::get_bounds },
    { "get_modified",           &TextBuffer::get_modified },
    { "set_modified",           &TextBuffer::set_modified },
    { "delete_selection",       &TextBuffer::delete_selection },
    //{ "paste_clipboard",        &TextBuffer::paste_clipboard },
    //{ "copy_clipboard",         &TextBuffer::copy_clipboard },
    //{ "cut_clipboard",          &TextBuffer::cut_clipboard },
    { "get_selection_bounds",   &TextBuffer::get_selection_bounds },
    { "begin_user_action",      &TextBuffer::begin_user_action },
    { "end_user_action",        &TextBuffer::end_user_action },
#if 0
    { "add_selection_clipboard",        &TextBuffer:: },
    { "remove_selection_clipboard",        &TextBuffer:: },
    { "deserialize",        &TextBuffer:: },
    { "deserialize_get_can_create_tags",        &TextBuffer:: },
    { "deserialize_set_can_create_tags",        &TextBuffer:: },
    { "get_copy_target_list",        &TextBuffer:: },
    { "get_deserialize_formats",        &TextBuffer:: },
    { "get_paste_target_list",        &TextBuffer:: },
    { "get_serialize_formats",        &TextBuffer:: },
    { "register_deserialize_format",        &TextBuffer:: },
    { "register_deserialize_tagset",        &TextBuffer:: },
    { "register_serialize_format",        &TextBuffer:: },
    { "register_serialize_tagset",        &TextBuffer:: },
    { "gtk_text_buffer_serialize",        &TextBuffer:: },
    { "unregister_deserialize_format",        &TextBuffer:: },
    { "unregister_serialize_format",        &TextBuffer:: },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextBuffer, meth->name, meth->cb );
}


TextBuffer::TextBuffer( const Falcon::CoreClass* gen, const GtkTextBuffer* buf )
    :
    Gtk::CoreGObject( gen, (GObject*) buf )
{}


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
            tab = (GtkTextTagTable*) COREGOBJECT( i_tab )->getObject();
        }
    }

    MYSELF;
    GtkTextBuffer* buf = gtk_text_buffer_new( tab );
    self->setObject( (GObject*) buf );
}


/*#
    @method get_line_count GtkTextBuffer
    @brief Obtains the number of lines in the buffer.
    @return number of lines in the buffer

    This value is cached, so the function is very fast.
 */
FALCON_FUNC TextBuffer::get_line_count( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_buffer_get_line_count( (GtkTextBuffer*)_obj ) );
}


/*#
    @method get_char_count GtkTextBuffer
    @brief Gets the number of characters in the buffer.

    Note that characters and bytes are not the same, you can't e.g. expect the
    contents of the buffer in string form to be this many bytes long. The character
    count is cached, so this function is very fast.
 */
FALCON_FUNC TextBuffer::get_char_count( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_buffer_get_char_count( (GtkTextBuffer*)_obj ) );
}


/*#
    @method get_tag_table GtkTextBuffer
    @brief Get the GtkTextTagTable associated with this buffer.
    @return the buffer's tag table.
 */
FALCON_FUNC TextBuffer::get_tag_table( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkTextTagTable* tab = gtk_text_buffer_get_tag_table( (GtkTextBuffer*)_obj );
    vm->retval( new Gtk::TextTagTable( vm->findWKI( "GtkTextTagTable" )->asClass(), tab ) );
}


/*#
    @method insert GtkTextBuffer
    @brief Inserts len bytes of text at position iter.
    @param iter (GtkTextIter) a position in the buffer
    @param text some text in UTF-8 format
    @param len length of text, in bytes

    If len is -1, text must be nul-terminated and will be inserted in its entirety.

    Emits the "insert-text" signal; insertion actually occurs in the default handler
    for the signal. iter is invalidated when insertion occurs (because the buffer
    contents change), but the default signal handler revalidates it to point to the
    end of the inserted text.
 */
FALCON_FUNC TextBuffer::insert( VMARG )
{
    Gtk::ArgCheck1 args( vm, "GtkTextIter,S,I" );

    CoreGObject* o_iter = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,S,I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();
    char* txt = args.getCString( 1 );
    int len = args.getInteger( 2 );

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_insert( (GtkTextBuffer*)_obj, iter, txt, len );
}


/*#
    @method insert_at_cursor GtkTextBuffer
    @brief Simply calls insert(), using the current cursor position as the insertion point.
    @param text some text in UTF-8 format
    @param len length of text, in bytes
 */
FALCON_FUNC TextBuffer::insert_at_cursor( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,I" );

    char* txt = args.getCString( 0 );
    int len = args.getInteger( 1 );

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_insert_at_cursor( (GtkTextBuffer*)_obj, txt, len );
}


/*#
    @method insert_interactive GtkTextBuffer
    @brief Like insert(), but the insertion will not occur if iter is at a non-editable location in the buffer.
    @param iter (GtkTextIter) a position in the buffer
    @param text some text in UTF-8 format
    @param len length of text, in bytes
    @param default_editable (boolean)
    @return whether text was actually inserted

    Usually you want to prevent insertions at ineditable locations if the insertion
    results from a user action (is interactive).

    default_editable indicates the editability of text that doesn't have a tag
    affecting editability applied to it. Typically the result of gtk_text_view_get_editable()
    is appropriate here.
 */
FALCON_FUNC TextBuffer::insert_interactive( VMARG )
{
    Gtk::ArgCheck1 args( vm, "GtkTextIter,S,I,B" );

    CoreGObject* o_iter = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,S,I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();
    char* txt = args.getCString( 1 );
    int len = args.getInteger( 2 );
    bool dft_edit = args.getBoolean( 3 );

    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_buffer_insert_interactive( (GtkTextBuffer*)_obj,
            iter, txt, len, dft_edit ) );
}


/*#
    @method insert_interactive_at_cursor GtkTextBuffer
    @brief Calls gtk_text_buffer_insert_interactive() at the cursor position.
    @param text some text in UTF-8 format
    @param len length of text, in bytes
    @param default_editable (boolean) default editability of buffer
    @return whether text was actually inserted

    default_editable indicates the editability of text that doesn't have a tag affecting
    editability applied to it. Typically the result of gtk_text_view_get_editable()
    is appropriate here.
 */
FALCON_FUNC TextBuffer::insert_interactive_at_cursor( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,I,B" );

    char* txt = args.getCString( 0 );
    int len = args.getInteger( 1 );
    bool dft_edit = args.getBoolean( 2 );

    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_buffer_insert_interactive_at_cursor(
            (GtkTextBuffer*)_obj, txt, len, dft_edit ) );
}


/*#
    @method insert_range GtkTextBuffer
    @brief Copies text, tags, and pixbufs between start and end (the order of start and end doesn't matter) and inserts the copy at iter.
    @param iter (GtkTextIter) a position in buffer
    @param start (GtkTextIter) a position in a GtkTextBuffer
    @param end (GtkTextIter) another position in the same buffer as start

    Used instead of simply getting/inserting text because it preserves images and tags.
    If start and end are in a different buffer from buffer, the two buffers must share
    the same tag table.

    Implemented via emissions of the insert_text and apply_tag signals, so expect those.
 */
FALCON_FUNC TextBuffer::insert_range( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,GtkTextIter,GtkTextIter" );

    CoreGObject* o_iter = args.getCoreGObject( 0 );
    CoreGObject* o_start = args.getCoreGObject( 1 );
    CoreGObject* o_end = args.getCoreGObject( 2 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_iter, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter,GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_insert_range( (GtkTextBuffer*)_obj, iter, start, end );
}


/*#
    @method insert_range_interactive GtkTextBuffer
    @brief Like insert(), but the insertion will not occur if iter is at a non-editable location in the buffer.
    @param iter (GtkTextIter) a position in buffer
    @param start (GtkTextIter) a position in a GtkTextBuffer
    @param end (GtkTextIter) another position in the same buffer as start
    @param default_editable (boolean) default editability of buffer
    @return whether an insertion was possible at iter

    Usually you want to prevent insertions at ineditable locations if the insertion
    results from a user action (is interactive).

    default_editable indicates the editability of text that doesn't have a tag affecting
    editability applied to it. Typically the result of gtk_text_view_get_editable()
    is appropriate here.
 */
FALCON_FUNC TextBuffer::insert_range_interactive( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,GtkTextIter,GtkTextIter,B" );

    CoreGObject* o_iter = args.getCoreGObject( 0 );
    CoreGObject* o_start = args.getCoreGObject( 1 );
    CoreGObject* o_end = args.getCoreGObject( 2 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_iter, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter,GtkTextIter,B" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();
    bool dft_edit = args.getBoolean( 3 );

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_insert_range_interactive( (GtkTextBuffer*)_obj, iter, start, end, dft_edit );
}


//FALCON_FUNC TextBuffer::insert_with_tags( VMARG );


//FALCON_FUNC TextBuffer::insert_with_tags_by_name( VMARG );


/*#
    @method delete GtkTextBuffer
    @brief Deletes text between start and end.
    @param start a position in buffer
    @param end another position in buffer

    The order of start and end is not actually relevant; delete() will reorder them.
    This function actually emits the "delete-range" signal, and the default handler
    of that signal deletes the text. Because the buffer is modified, all outstanding
    iterators become invalid after calling this function; however, the start and end
    will be re-initialized to point to the location where text was deleted.
 */
FALCON_FUNC TextBuffer::delete_( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,GtkTextIter" );

    CoreGObject* o_start = args.getCoreGObject( 0 );
    CoreGObject* o_end = args.getCoreGObject( 1 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter" );
#endif
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_delete( (GtkTextBuffer*)_obj, start, end );
}


/*#
    @method delete_interactive GtkTextBuffer
    @brief Deletes all editable text in the given range.
    @param start a position in buffer
    @param end another position in buffer
    @param default_editable whether the buffer is editable by default
    @return whether some text was actually deleted

    Calls delete() for each editable sub-range of [start,end]. start and end are
    revalidated to point to the location of the last deleted range, or left untouched
    if no text was deleted.
 */
FALCON_FUNC TextBuffer::delete_interactive( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,GtkTextIter,B" );

    CoreGObject* o_start = args.getCoreGObject( 0 );
    CoreGObject* o_end = args.getCoreGObject( 1 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter,B" );
#endif
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();
    gboolean dft_edit = args.getBoolean( 2 );

    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_buffer_delete_interactive( (GtkTextBuffer*)_obj,
            start, end, dft_edit ) );
}


/*#
    @method backspace GtkTextBuffer
    @brief Performs the appropriate action as if the user hit the delete key with the cursor at the position specified by iter.
    @param iter a position in buffer
    @param interactive whether the deletion is caused by user interaction
    @param default_editable whether the buffer is editable by default
    @return true if the buffer was modified

    In the normal case a single character will be deleted, but when combining accents
    are involved, more than one character can be deleted, and when precomposed character
    and accent combinations are involved, less than one character will be deleted.

    Because the buffer is modified, all outstanding iterators become invalid after
    calling this function; however, the iter will be re-initialized to point to the
    location where text was deleted.
 */
FALCON_FUNC TextBuffer::backspace( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,B,B" );

    CoreGObject* o_iter = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,B,B" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();
    gboolean inter = args.getBoolean( 1 );
    gboolean dft_edit = args.getBoolean( 2 );

    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_buffer_backspace( (GtkTextBuffer*)_obj,
            iter, inter, dft_edit ) );
}


/*#
    @method set_text GtkTextBuffer
    @brief Deletes current contents of buffer, and inserts text instead.
    @param text UTF-8 text to insert
    @param len length of text in bytes

    If len is -1, text must be nul-terminated. text must be valid UTF-8.
 */
FALCON_FUNC TextBuffer::set_text( VMARG )
{
    Gtk::ArgCheck1 args ( vm, "S,I" );

    char* txt = args.getCString( 0 );
    int len = args.getInteger( 1 );

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_set_text( (GtkTextBuffer*)_obj, txt, len );
}


/*#
    @method get_text GtkTextBuffer
    @brief Returns the text in the range [start,end).
    @param start start of range
    @param end end of range
    @param include_hidden_chars whether to include invisible text
    @return an UTF-8 string

    Excludes undisplayed text (text marked with tags that set the invisibility
    attribute) if include_hidden_chars is FALSE. Does not include characters
    representing embedded images, so byte and character indexes into the returned
    string do not correspond to byte and character indexes into the buffer.
    Contrast with gtk_text_buffer_get_slice().
 */
FALCON_FUNC TextBuffer::get_text( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,GtkTextIter,B" );

    CoreGObject* o_start = args.getCoreGObject( 0 );
    CoreGObject* o_end = args.getCoreGObject( 1 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter,B" );
#endif
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();
    gboolean hide = args.getBoolean( 2 );

    MYSELF;
    GET_OBJ( self );
    gchar* res = gtk_text_buffer_get_text( (GtkTextBuffer*)_obj, start, end, hide );
    if ( res )
    {
        String* s = new String( res );
        s->bufferize();
        vm->retval( s );
        g_free( res );
    }
    else
        vm->retnil();
}


/*#
    @method get_slice GtkTextBuffer
    @brief Returns the text in the range [start,end).
    @param start start of range
    @param end end of range
    @param include_hidden_chars whether to include invisible text
    @return an UTF-8 string

    Excludes undisplayed text (text marked with tags that set the invisibility attribute)
    if include_hidden_chars is FALSE. The returned string includes a 0xFFFC character
    whenever the buffer contains embedded images, so byte and character indexes into
    the returned string do correspond to byte and character indexes into the buffer.
    Contrast with gtk_text_buffer_get_text(). Note that 0xFFFC can occur in normal
    text as well, so it is not a reliable indicator that a pixbuf or widget is in the buffer.
 */
FALCON_FUNC TextBuffer::get_slice( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,GtkTextIter,B" );

    CoreGObject* o_start = args.getCoreGObject( 0 );
    CoreGObject* o_end = args.getCoreGObject( 1 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter,B" );
#endif
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();
    gboolean hide = args.getBoolean( 2 );

    MYSELF;
    GET_OBJ( self );
    gchar* res = gtk_text_buffer_get_slice( (GtkTextBuffer*)_obj, start, end, hide );
    if ( res )
    {
        String* s = new String( res );
        s->bufferize();
        vm->retval( s );
        g_free( res );
    }
    else
        vm->retnil();
}


/*#
    @method insert_pixbuf GtkTextBuffer
    @brief Inserts an image into the text buffer at iter.
    @param iter location to insert the pixbuf
    @param pixbuf a GdkPixbuf

    The image will be counted as one character in character counts, and when obtaining
    the buffer contents as a string, will be represented by the Unicode "object
    replacement character" 0xFFFC. Note that the "slice" variants for obtaining
    portions of the buffer as a string include this character for pixbufs, but the
    "text" variants do not. e.g. see gtk_text_buffer_get_slice() and gtk_text_buffer_get_text().
 */
FALCON_FUNC TextBuffer::insert_pixbuf( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,GdkPixbuf" );

    CoreGObject* o_iter = args.getCoreGObject( 0 );
    CoreGObject* o_pix = args.getCoreGObject( 1 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_iter, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_pix, GdkPixbuf ) )
        throw_inv_params( "GtkTextIter,GdkPixbuf" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();
    GdkPixbuf* pix = (GdkPixbuf*) o_pix->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_insert_pixbuf( (GtkTextBuffer*)_obj, iter, pix );
}


//FALCON_FUNC TextBuffer::insert_child_anchor( VMARG );

//FALCON_FUNC TextBuffer::create_child_anchor( VMARG );


/*#
    @method create_mark GtkTextBuffer
    @brief Creates a mark at position where.
    @param mark_name name for mark, or nil
    @param where (GtkTextIter) location to place mark
    @param left_gravity (boolean) whether the mark has left gravity

    If mark_name is NULL, the mark is anonymous; otherwise, the mark can be retrieved
    by name using gtk_text_buffer_get_mark(). If a mark has left gravity, and text
    is inserted at the mark's current location, the mark will be moved to the left
    of the newly-inserted text. If the mark has right gravity (left_gravity = FALSE),
    the mark will end up on the right of newly-inserted text. The standard left-to-right
    cursor is a mark with right gravity (when you type, the cursor stays on the right
    side of the text you're typing).

    The caller of this function does not own a reference to the returned GtkTextMark,
    so you can ignore the return value if you like. Marks are owned by the buffer and
    go away when the buffer does.

    Emits the "mark-set" signal as notification of the mark's initial placement.
 */
FALCON_FUNC TextBuffer::create_mark( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,GtkTextIter,B" );

    char* name = args.getCString( 0, false );
    CoreGObject* o_iter = args.getCoreGObject( 1 );
    gboolean gravity = args.getBoolean( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_iter, GtkTextIter ) )
        throw_inv_params( "S,GtkTextIter,B" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();

    MYSELF;
    GET_OBJ( self );
    GtkTextMark* mk = gtk_text_buffer_create_mark( (GtkTextBuffer*)_obj,
            name, iter, gravity );
    vm->retval( new Gtk::TextMark( vm->findWKI( "GtkTextIter" )->asClass(), mk ) );
}


/*#
    @method move_mark GtkTextBuffer
    @brief Moves mark to the new location where.
    @param mark a GtkTextMark
    @param where new location for mark in buffer

    Emits the "mark-set" signal as notification of the move.
 */
FALCON_FUNC TextBuffer::move_mark( VMARG )
{
    Item* i_mk = vm->param( 0 );
    Item* i_iter = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mk || i_mk->isNil() || !i_mk->isObject()
        || !IS_DERIVED( i_mk, GtkTextMark )
        || !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextMark,GtkTextIter" );
#endif
    GtkTextMark* mk = (GtkTextMark*) COREGOBJECT( i_mk )->getObject();
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_move_mark ( (GtkTextBuffer*)_obj, mk, iter );
}


/*#
    @method move_mark_by_name GtkTextBuffer
    @brief Moves the mark named name (which must exist) to location where.
    @param name name of a mark
    @param where new location for mark

    See move_mark() for details.
 */
FALCON_FUNC TextBuffer::move_mark_by_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,GtkTextIter" );

    char* nam = args.getCString( 0 );
    CoreGObject* o_iter = args.getCoreGObject( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_iter, GtkTextIter ) )
        throw_inv_params( "S,GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_move_mark_by_name( (GtkTextBuffer*)_obj, nam, iter );
}

/*#
    @method add_mark GtkTextBuffer
    @brief Adds the mark at position where.
    @param mark the mark to add
    @param where location to place mark

    The mark must not be added to another buffer, and if its name is not NULL then
    there must not be another mark in the buffer with the same name.

    Emits the "mark-set" signal as notification of the mark's initial placement.
 */
FALCON_FUNC TextBuffer::add_mark( VMARG )
{
    Item* i_mk = vm->param( 0 );
    Item* i_iter = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mk || i_mk->isNil() || !i_mk->isObject()
        || !IS_DERIVED( i_mk, GtkTextMark )
        || !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextMark,GtkTextIter" );
#endif
    GtkTextMark* mk = (GtkTextMark*) COREGOBJECT( i_mk )->getObject();
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_add_mark( (GtkTextBuffer*)_obj, mk, iter );
}


/*#
    @method delete_mark GtkTextBuffer
    @brief Deletes mark, so that it's no longer located anywhere in the buffer.
    @param amrk a GtkTextMark in buffer

    Most operations on mark become invalid, until the mark gets added to a buffer again
    with gtk_text_buffer_add_mark(). Use gtk_text_mark_get_deleted() to find out
    if a mark has been removed from its buffer. The "mark-deleted" signal will be
    emitted as notification after the mark is deleted.
 */
FALCON_FUNC TextBuffer::delete_mark( VMARG )
{
    Item* i_mk = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mk || i_mk->isNil() || !i_mk->isObject()
        || !IS_DERIVED( i_mk, GtkTextMark ) )
        throw_inv_params( "GtkTextMark" );
#endif
    GtkTextMark* mk = (GtkTextMark*) COREGOBJECT( i_mk )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_delete_mark( (GtkTextBuffer*)_obj, mk );
}


/*#
    @method delete_mark_by_name GtkTextBuffer
    @brief Deletes the mark named name; the mark must exist.
    @param name name of a mark in buffer

    See delete_mark() for details.
 */
FALCON_FUNC TextBuffer::delete_mark_by_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );

    char* nam = args.getCString( 0 );

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_delete_mark_by_name( (GtkTextBuffer*)_obj, nam );
}


/*#
    @method get_mark GtkTextBuffer
    @brief Returns the mark named name in buffer buffer, or nil if no such mark exists in the buffer.
    @param name a mark name
    @return a GtkTextMark, or nil.
 */
FALCON_FUNC TextBuffer::get_mark( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );

    char* name = args.getCString( 0 );

    MYSELF;
    GET_OBJ( self );
    GtkTextMark* mk = gtk_text_buffer_get_mark( (GtkTextBuffer*)_obj, name );
    if ( mk )
        vm->retval( new Gtk::TextMark( vm->findWKI( "GtkTextMark" )->asClass(), mk ) );
    else
        vm->retnil();
}


/*#
    @method get_insert GtkTextBuffer
    @brief Returns the mark that represents the cursor (insertion point).
    @return (GtkTextMark) insertion point mark.

    Equivalent to calling gtk_text_buffer_get_mark() to get the mark named "insert",
    but very slightly more efficient, and involves less typing.
 */
FALCON_FUNC TextBuffer::get_insert( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkTextMark* mk = gtk_text_buffer_get_insert( (GtkTextBuffer*)_obj );
    vm->retval( new Gtk::TextMark( vm->findWKI( "GtkTextMark" )->asClass(), mk ) );
}


/*#
    @method get_selection_bound GtkTextBuffer
    @brief Returns the mark that represents the selection bound.
    @return (GtkTextMark) selection bound mark.

    Equivalent to calling get_mark() to get the mark named "selection_bound",
    but very slightly more efficient, and involves less typing.

    The currently-selected text in buffer is the region between the "selection_bound"
    and "insert" marks. If "selection_bound" and "insert" are in the same place, then
    there is no current selection. get_selection_bounds() is another
    convenient function for handling the selection, if you just want to know whether
    there's a selection and what its bounds are.
 */
FALCON_FUNC TextBuffer::get_selection_bound( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkTextMark* mk = gtk_text_buffer_get_selection_bound( (GtkTextBuffer*)_obj );
    vm->retval( new Gtk::TextMark( vm->findWKI( "GtkTextMark" )->asClass(), mk ) );
}


/*#
    @method get_has_selection GtkTextBuffer
    @brief Indicates whether the buffer has some text currently selected.
    @return true if the there is text selected
 */
FALCON_FUNC TextBuffer::get_has_selection( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_buffer_get_has_selection( (GtkTextBuffer*)_obj ) );
}


/*#
    @method place_cursor GtkTextBuffer
    @brief This function moves the "insert" and "selection_bound" marks simultaneously.
    @param where (GtkTextIter) where to put the cursor

    If you move them to the same place in two steps with move_mark(),
    you will temporarily select a region in between their old and new locations,
    which can be pretty inefficient since the temporarily-selected region will force
    stuff to be recalculated. This function moves them as a unit, which can be optimized.
 */
FALCON_FUNC TextBuffer::place_cursor( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_place_cursor( (GtkTextBuffer*)_obj, iter );
}


/*#
    @method select_range GtkTextBuffer
    @brief This function moves the "insert" and "selection_bound" marks simultaneously.
    @param ins (GtkTextIter) where to put the "insert" mark
    @param bound (GtkTextIter) where to put the "selection_bound" mark

    If you move them in two steps with move_mark(), you will temporarily select a
    region in between their old and new locations, which can be pretty inefficient
    since the temporarily-selected region will force stuff to be recalculated.
    This function moves them as a unit, which can be optimized.
 */
FALCON_FUNC TextBuffer::select_range( VMARG )
{
    Item* i_ins = vm->param( 0 );
    Item* i_bnd = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ins || i_ins->isNil() || !i_ins->isObject()
        || !IS_DERIVED( i_ins, GtkTextIter )
        || !i_bnd || i_bnd->isNil() || !i_bnd->isObject()
        || !IS_DERIVED( i_bnd, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter" );
#endif
    GtkTextIter* ins = (GtkTextIter*) COREGOBJECT( i_ins )->getObject();
    GtkTextIter* bnd = (GtkTextIter*) COREGOBJECT( i_bnd )->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_select_range( (GtkTextBuffer*)_obj, ins, bnd );
}


/*#
    @method apply_tag GtkTextBuffer
    @brief Emits the "apply-tag" signal on buffer.
    @param tag a GtkTextTag
    @param start one bound of range to be tagged
    @param end other bound of range to be tagged

    The default handler for the signal applies tag to the given range.
    start and end do not have to be in order.
 */
FALCON_FUNC TextBuffer::apply_tag( VMARG )
{
    Item* i_tag = vm->param( 0 );
    Item* i_start = vm->param( 1 );
    Item* i_end = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tag || i_tag->isNil() || !i_tag->isObject()
        || !IS_DERIVED( i_tag, GtkTextTag )
        || !i_start || i_start->isNil() || !i_start->isObject()
        || !IS_DERIVED( i_start, GtkTextIter )
        || !i_end || i_end->isNil() || !i_end->isObject()
        || !IS_DERIVED( i_end, GtkTextIter ) )
        throw_inv_params( "GtkTextTag,GtkTextIter,GtkTextIter" );
#endif
    GtkTextTag* tag = (GtkTextTag*) COREGOBJECT( i_tag )->getObject();
    GtkTextIter* start = (GtkTextIter*) COREGOBJECT( i_start )->getObject();
    GtkTextIter* end = (GtkTextIter*) COREGOBJECT( i_end )->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_apply_tag( (GtkTextBuffer*)_obj, tag, start, end );
}


/*#
    @method remove_tag GtkTextBuffer
    @brief Emits the "remove-tag" signal.
    @param tag a GtkTextTag
    @param start one bound of range to be untagged
    @param end other bound of range to be untagged

    The default handler for the signal removes all occurrences of tag from the given
    range. start and end don't have to be in order.
 */
FALCON_FUNC TextBuffer::remove_tag( VMARG )
{
    Item* i_tag = vm->param( 0 );
    Item* i_start = vm->param( 1 );
    Item* i_end = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tag || i_tag->isNil() || !i_tag->isObject()
        || !IS_DERIVED( i_tag, GtkTextTag )
        || !i_start || i_start->isNil() || !i_start->isObject()
        || !IS_DERIVED( i_start, GtkTextIter )
        || !i_end || i_end->isNil() || !i_end->isObject()
        || !IS_DERIVED( i_end, GtkTextIter ) )
        throw_inv_params( "GtkTextTag,GtkTextIter,GtkTextIter" );
#endif
    GtkTextTag* tag = (GtkTextTag*) COREGOBJECT( i_tag )->getObject();
    GtkTextIter* start = (GtkTextIter*) COREGOBJECT( i_start )->getObject();
    GtkTextIter* end = (GtkTextIter*) COREGOBJECT( i_end )->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_remove_tag( (GtkTextBuffer*)_obj, tag, start, end );
}


/*#
    @method apply_tag_by_name GtkTextBuffer
    @brief Calls gtk_text_tag_table_lookup() on the buffer's tag table to get a GtkTextTag, then calls gtk_text_buffer_apply_tag().
    @param name name of a named GtkTextTag
    @param start one bound of range to be tagged
    @param end other bound of range to be tagged
 */
FALCON_FUNC TextBuffer::apply_tag_by_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,GtkTextIter,GtkTextIter" );

    char* name = args.getCString( 0 );
    CoreGObject* o_start = args.getCoreGObject( 1 );
    CoreGObject* o_end = args.getCoreGObject( 2 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "S,GtkTextIter,GtkTextIter" );
#endif
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_apply_tag_by_name( (GtkTextBuffer*)_obj, name, start, end );
}


/*#
    @method remove_tag_by_name GtkTextBuffer
    @brief Calls gtk_text_tag_table_lookup() on the buffer's tag table to get a GtkTextTag, then calls gtk_text_buffer_remove_tag().
    @param name name of a named GtkTextTag
    @param start one bound of range to be untagged
    @param end other bound of range to be untagged
 */
FALCON_FUNC TextBuffer::remove_tag_by_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,GtkTextIter,GtkTextIter" );

    char* name = args.getCString( 0 );
    CoreGObject* o_start = args.getCoreGObject( 1 );
    CoreGObject* o_end = args.getCoreGObject( 2 );
#ifndef NO_PARAMETER_CHECK
    if (   !CoreObject_IS_DERIVED( o_start, GtkTextIter )
        || !CoreObject_IS_DERIVED( o_end, GtkTextIter ) )
        throw_inv_params( "S,GtkTextIter,GtkTextIter" );
#endif
    GtkTextIter* start = (GtkTextIter*) o_start->getObject();
    GtkTextIter* end = (GtkTextIter*) o_end->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_remove_tag_by_name( (GtkTextBuffer*)_obj, name, start, end );
}


/*#
    @method remove_all_tags GtkTextBuffer
    @brief Removes all tags in the range between start and end.
    @param start one bound of range to be untagged
    @param end other bound of range to be untagged

    Be careful with this function; it could remove tags added in code unrelated to
    the code you're currently writing. That is, using this function is probably a
    bad idea if you have two or more unrelated code sections that add tags.
 */
FALCON_FUNC TextBuffer::remove_all_tags( VMARG )
{
    Item* i_start = vm->param( 0 );
    Item* i_end = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_start || i_start->isNil() || !i_start->isObject()
        || !IS_DERIVED( i_start, GtkTextIter )
        || !i_end || i_end->isNil() || !i_end->isObject()
        || !IS_DERIVED( i_end, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,GtkTextIter" );
#endif
    GtkTextIter* start = (GtkTextIter*) COREGOBJECT( i_start )->getObject();
    GtkTextIter* end = (GtkTextIter*) COREGOBJECT( i_end )->getObject();

    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_remove_all_tags( (GtkTextBuffer*)_obj, start, end );
}


//FALCON_FUNC TextBuffer::create_tag( VMARG );


/*#
    @method get_iter_at_line_offset GtkTextBuffer
    @brief Obtains an iterator pointing to char_offset within the given line.
    @param line_number line number counting from 0
    @param line_offset char offset from start of line
    @return (GtkTextIter)

    The char_offset must exist, offsets off the end of the line are not allowed.
    Note characters, not bytes; UTF-8 may encode one character as multiple bytes.
 */
FALCON_FUNC TextBuffer::get_iter_at_line_offset( VMARG )
{
    Item* i_num = vm->param( 0 );
    Item* i_off = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if (   !i_num || i_num->isNil() || !i_num->isInteger()
        || !i_off || i_off->isNil() || !i_off->isInteger() )
        throw_inv_params( "I,I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_get_iter_at_line_offset( (GtkTextBuffer*)_obj, iter,
            i_num->asInteger(), i_off->asInteger() );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


/*#
    @method get_iter_at_offset GtkTextBuffer
    @brief Initializes iter to a position char_offset chars from the start of the entire buffer.
    @param char_offset char offset from start of buffer, counting from 0, or -1
    @return (GtkTextIter)

    If char_offset is -1 or greater than the number of characters in the buffer,
    iter is initialized to the end iterator, the iterator one past the last valid
    character in the buffer.
 */
FALCON_FUNC TextBuffer::get_iter_at_offset( VMARG )
{
    Item* i_off = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_off || i_off->isNil() || !i_off->isInteger() )
        throw_inv_params( "I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_get_iter_at_offset( (GtkTextBuffer*)_obj, iter, i_off->asInteger() );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


/*#
    @method get_iter_at_line GtkTextBuffer
    @brief Initializes iter to the start of the given line.
    @param line_number line number counting from 0
    @return (GtkTextIter)
 */
FALCON_FUNC TextBuffer::get_iter_at_line( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_get_iter_at_line( (GtkTextBuffer*)_obj, iter, i_num->asInteger() );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


/*#
    @method get_iter_at_line_index GtkTextBuffer
    @brief Obtains an iterator pointing to byte_index within the given line.
    @param line_number line number counting from 0
    @param byte_offset byte index from start of line
    @return (GtkTextIter)

    byte_index must be the start of a UTF-8 character, and must not be beyond the
    end of the line. Note bytes, not characters; UTF-8 may encode one character
    as multiple bytes.
 */
FALCON_FUNC TextBuffer::get_iter_at_line_index( VMARG )
{
    Item* i_num = vm->param( 0 );
    Item* i_off = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if (   !i_num || i_num->isNil() || !i_num->isInteger()
        || !i_off || i_off->isNil() || !i_off->isInteger() )
        throw_inv_params( "I,I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_get_iter_at_line_index( (GtkTextBuffer*)_obj, iter,
            i_num->asInteger(), i_off->asInteger() );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


//FALCON_FUNC TextBuffer::get_iter_at_mark( VMARG );

//FALCON_FUNC TextBuffer::get_iter_at_child_anchor( VMARG );


/*#
    @method get_start_iter GtkTextBuffer
    @brief Initialize iter with the first position in the text buffer.
    @return (GtkTextIter)

    This is the same as using gtk_text_buffer_get_iter_at_offset() to get the iter
    at character offset 0.
 */
FALCON_FUNC TextBuffer::get_start_iter( VMARG )
{
    NO_ARGS
    GtkTextIter* iter = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_get_start_iter( (GtkTextBuffer*)_obj, iter );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


/*#
    @method get_end_iter GtkTextBuffer
    @brief Initializes iter with the "end iterator," one past the last valid character in the text buffer.
    @return (GtkTextIter)

    If dereferenced with gtk_text_iter_get_char(), the end iterator has a character
    value of 0. The entire buffer lies in the range from the first position in the
    buffer (call gtk_text_buffer_get_start_iter() to get character position 0) to
    the end iterator.
 */
FALCON_FUNC TextBuffer::get_end_iter( VMARG )
{
    NO_ARGS
    GtkTextIter* iter = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_get_end_iter( (GtkTextBuffer*)_obj, iter );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


/*#
    @method get_bounds GtkTextBuffer
    @brief Retrieves the first and last iterators in the buffer, i.e. the entire buffer lies within the range [start,end].
    @return [ GtkTextIter, GtkTextIter ]
 */
FALCON_FUNC TextBuffer::get_bounds( VMARG )
{
    NO_ARGS
    GtkTextIter* start = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    GtkTextIter* end = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_get_bounds( (GtkTextBuffer*)_obj, start, end );

    CoreArray* arr = new CoreArray( 2 );
    Item* wki = vm->findWKI( "GtkTextIter" );
    arr->append( new Gtk::TextIter( wki->asClass(), start ) );
    arr->append( new Gtk::TextIter( wki->asClass(), end ) );
    vm->retval( arr );
}


/*#
    @method get_modified GtkTextBuffer
    @brief Indicates whether the buffer has been modified since the last call to set_modified() set the modification flag to false.
    @return true if the buffer has been modified
    Used for example to enable a "save" function in a text editor.
 */
FALCON_FUNC TextBuffer::get_modified( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_buffer_get_modified( (GtkTextBuffer*)_obj ) );
}


/*#
    @method set_modified GtkTextBuffer
    @brief Used to keep track of whether the buffer has been modified since the last time it was saved.
    @param setting modification flag setting

    Whenever the buffer is saved to disk, call set_modified (buffer, FALSE).
    When the buffer is modified, it will automatically toggled on the modified bit again.
    When the modified bit flips, the buffer emits a "modified-changed" signal.
 */
FALCON_FUNC TextBuffer::set_modified( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_set_modified( (GtkTextBuffer*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method delete_selection GtkTextBuffer
    @brief Deletes the range between the "insert" and "selection_bound" marks, that is, the currently-selected text.
    @param interactive whether the deletion is caused by user interaction
    @param default_editable whether the buffer is editable by default
    @return whether there was a non-empty selection to delete

    If interactive is true, the editability of the selection will be considered
    (users can't delete uneditable text).
 */
FALCON_FUNC TextBuffer::delete_selection( VMARG )
{
    Item* i_inter = vm->param( 0 );
    Item* i_dft_edit = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_inter || i_inter->isNil() || !i_inter->isBoolean()
        || !i_dft_edit || i_dft_edit->isNil() || !i_dft_edit->isBoolean() )
        throw_inv_params( "B,B" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_buffer_delete_selection( (GtkTextBuffer*)_obj,
        i_inter->asBoolean() ? TRUE : FALSE, i_dft_edit->asBoolean() ? TRUE : FALSE ) );
}


//FALCON_FUNC TextBuffer::paste_clipboard( VMARG );

//FALCON_FUNC TextBuffer::copy_clipboard( VMARG );

//FALCON_FUNC TextBuffer::cut_clipboard( VMARG );


/*#
    @method get_selection_bounds GtkTextBuffer
    @brief Returns true if some text is selected; places the bounds of the selection in start and end (if the selection has length 0, then start and end are filled in with the same value).
    @return [ boolean (whether the selection has nonzero length ), (GtkTextIter) start, (GtkTextIter) end ]

    start and end will be in ascending order. If start and end are nil, then they
    are not filled in, but the return value still indicates whether text is selected.
 */
FALCON_FUNC TextBuffer::get_selection_bounds( VMARG )
{
    NO_ARGS
    GtkTextIter* start = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    GtkTextIter* end = (GtkTextIter*) malloc( sizeof( GtkTextIter ) );
    MYSELF;
    GET_OBJ( self );
    gboolean res = gtk_text_buffer_get_selection_bounds( (GtkTextBuffer*)_obj, start, end );

    CoreArray* arr = new CoreArray( 3 );
    arr->append( (bool) res );
    if ( res )
    {
        Item* wki = vm->findWKI( "GtkTextIter" );
        arr->append( new Gtk::TextIter( wki->asClass(), start ) );
        arr->append( new Gtk::TextIter( wki->asClass(), end ) );
    }
    else
    {
        free( start );
        free( end );
        arr->append( Item() );
        arr->append( Item() );
    }
    vm->retval( arr );
}


/*#
    @method begin_user_action GtkTextBuffer
    @brief Called to indicate that the buffer operations between here and a call to gtk_text_buffer_end_user_action() are part of a single user-visible operation.

    The operations between begin_user_action() and end_user_action() can then be grouped
    when creating an undo stack. GtkTextBuffer maintains a count of calls to
    begin_user_action() that have not been closed with a call to end_user_action(),
    and emits the "begin-user-action" and "end-user-action" signals only for the outermost
    pair of calls. This allows you to build user actions from other user actions.

    The "interactive" buffer mutation functions, such as insert_interactive(), automatically
    call begin/end user action around the buffer operations they perform, so there's
    no need to add extra calls if you user action consists solely of a single call
    to one of those functions.
 */
FALCON_FUNC TextBuffer::begin_user_action( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_begin_user_action( (GtkTextBuffer*)_obj );
}


/*#
    @method end_user_action GtkTextBuffer
    @brief Should be paired with a call to begin_user_action(). See that function for a full explanation.
 */
FALCON_FUNC TextBuffer::end_user_action( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_text_buffer_end_user_action( (GtkTextBuffer*)_obj );
}


//FALCON_FUNC TextBuffer::add_selection_clipboard( VMARG );

//FALCON_FUNC TextBuffer::remove_selection_clipboard( VMARG );

//FALCON_FUNC TextBuffer::deserialize( VMARG );

//FALCON_FUNC TextBuffer::deserialize_get_can_create_tags( VMARG );

//FALCON_FUNC TextBuffer::deserialize_set_can_create_tags( VMARG );

//FALCON_FUNC TextBuffer::get_copy_target_list( VMARG );

//FALCON_FUNC TextBuffer::get_deserialize_formats( VMARG );

//FALCON_FUNC TextBuffer::get_paste_target_list( VMARG );

//FALCON_FUNC TextBuffer::get_serialize_formats( VMARG );

//FALCON_FUNC TextBuffer::register_deserialize_format( VMARG );

//FALCON_FUNC TextBuffer::register_deserialize_tagset( VMARG );

//FALCON_FUNC TextBuffer::register_serialize_format( VMARG );

//FALCON_FUNC TextBuffer::register_serialize_tagset( VMARG );

//FALCON_FUNC TextBuffer::gtk_text_buffer_serialize( VMARG );

//FALCON_FUNC TextBuffer::unregister_deserialize_format( VMARG );

//FALCON_FUNC TextBuffer::unregister_serialize_format( VMARG );


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
