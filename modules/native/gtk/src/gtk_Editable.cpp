/**
 *  \file gtk_Editable.cpp
 */

#include "gtk_Editable.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {


/**
 *  \brief interface loader
 */
void Editable::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "select_region",          &Editable::select_region },
    { "get_selection_bounds",   &Editable::get_selection_bounds },
    { "insert_text",            &Editable::insert_text },
    { "delete_text",            &Editable::delete_text },
    { "get_chars",              &Editable::get_chars },
    { "cut_clipboard",          &Editable::cut_clipboard },
    { "copy_clipboard",         &Editable::copy_clipboard },
    { "paste_clipboard",        &Editable::paste_clipboard },
    { "delete_selection",       &Editable::delete_selection },
    { "set_position",           &Editable::set_position },
    { "get_position",           &Editable::get_position },
    { "set_editable",           &Editable::set_editable },
    { "get_editable",           &Editable::get_editable },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


/*#
    @class GtkEditable
    @brief Interface for text-editing widgets

    GtkEditable is implemented by GtkEntry, GtkText, GtkOldEditable and GtkSpinButton.

    The GtkEditable interface is an interface which should be implemented by text
    editing widgets, such as GtkEntry and GtkText. It contains functions for generically
    manipulating an editable widget, a large number of action signals used for key
    bindings, and several signals that an application can connect to to modify the
    behavior of a widget.

    As an example of the latter usage, by connecting the following handler to "insert_text",
    an application can convert all entry into a widget into uppercase.
 */


/*#
    @method select_region GtkEditable
    @brief Selects a region of text.
    @param start_pos start of region
    @param end_pos end of region

    The characters that are selected are those characters at positions from
    start_pos up to, but not including end_pos. If end_pos is negative, then the
    the characters selected are those characters from start_pos to the end of the text.

    Note that positions are specified in characters, not bytes.
 */
FALCON_FUNC Editable::select_region( VMARG )
{
    Gtk::ArgCheck0 args( vm, "I,I" );

    gint start = args.getInteger( 0 );
    gint end = args.getInteger( 1 );

    MYSELF;
    GET_OBJ( self );
    gtk_editable_select_region( (GtkEditable*)_obj, start, end );
}


/*#
    @method get_selection_bounds GtkEditable
    @brief Retrieves the selection bound of the editable.
    @return [ result (boolean), start_pos, end_pos ]. Result is true if an area is selected, false otherwise.

    start_pos will be filled with the start of the selection and end_pos with end.
    If no text was selected both will be identical and FALSE will be returned.

    Note that positions are specified in characters, not bytes.
 */
FALCON_FUNC Editable::get_selection_bounds( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint start, end;
    gboolean res = gtk_editable_get_selection_bounds( (GtkEditable*)_obj, &start, &end );
    CoreArray* arr = new CoreArray( 3 );
    arr->append( (bool) res );
    arr->append( start );
    arr->append( end );
    vm->retval( arr );
}


/*#
    @method insert_text GtkEditable
    @brief Inserts new_text into the contents of the widget, at position position.
    @param new_text the text to append
    @optparam position location of the position text will be inserted at. (default 0)
    @return position

    Note that the position is in characters, not in bytes. The function updates
    position to point after the newly inserted text.
 */
FALCON_FUNC Editable::insert_text( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S[,I]" );

    const char* txt = args.getCString( 0 );
    gint pos = args.getInteger( 1, false );

    MYSELF;
    GET_OBJ( self );
    gtk_editable_insert_text( (GtkEditable*)_obj, txt, -1, &pos );
    vm->retval( pos );
}


/*#
    @method delete_text GtkEditable
    @brief Deletes a sequence of characters.
    @param start_pos start of text
    @param end_pos end of text

    The characters that are deleted are those characters at positions from start_pos
    up to, but not including end_pos. If end_pos is negative, then the the characters
    deleted are those from start_pos to the end of the text.

    Note that the positions are specified in characters, not bytes.
 */
FALCON_FUNC Editable::delete_text( VMARG )
{
    Gtk::ArgCheck0 args( vm, "I,I" );

    gint start = args.getInteger( 0 );
    gint end = args.getInteger( 1 );

    MYSELF;
    GET_OBJ( self );
    gtk_editable_delete_text( (GtkEditable*)_obj, start, end );
}


/*#
    @method get_chars GtkEditable
    @brief Retrieves a sequence of characters.
    @optparam start_pos start of text (default 0)
    @optparam end_pos end of text (default -1)
    @return contents of the widget as a string.

    The characters that are retrieved are those characters at positions from start_pos
    up to, but not including end_pos. If end_pos is negative, then the the characters
    retrieved are those characters from start_pos to the end of the text.

    Note that positions are specified in characters, not bytes.
 */
FALCON_FUNC Editable::get_chars( VMARG )
{
    Gtk::ArgCheck0 args( vm, "I,I" );

    gint start = args.getInteger( 0, false );
    bool wasNil = false;
    gint end = args.getInteger( 1, false, &wasNil );
    if ( wasNil )
        end = -1;

    MYSELF;
    GET_OBJ( self );
    gchar* txt = gtk_editable_get_chars( (GtkEditable*)_obj, start, end );
    String* s = new String( txt );
    s->bufferize();
    vm->retval( s );
    g_free( txt );
}


/*#
    @method cut_clipboard GtkEditable
    @brief Removes the contents of the currently selected content in the editable and puts it on the clipboard.
 */
FALCON_FUNC Editable::cut_clipboard( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_editable_cut_clipboard( (GtkEditable*)_obj );
}


/*#
    @method copy_clipboard GtkEditable
    @brief Copies the contents of the currently selected content in the editable and puts it on the clipboard.
 */
FALCON_FUNC Editable::copy_clipboard( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_editable_copy_clipboard( (GtkEditable*)_obj );
}


/*#
    @method paste_clipboard GtkEditable
    @brief Pastes the content of the clipboard to the current position of the cursor in the editable.
 */
FALCON_FUNC Editable::paste_clipboard( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_editable_paste_clipboard( (GtkEditable*)_obj );
}


/*#
    @method delete_selection GtkEditable
    @brief Deletes the currently selected text of the editable. This call doesn't do anything if there is no selected text.
 */
FALCON_FUNC Editable::delete_selection( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_editable_delete_selection( (GtkEditable*)_obj );
}


/*#
    @method set_position GtkEditable
    @brief Sets the cursor position in the editable to the given value.
    @param position the position of the cursor

    The cursor is displayed before the character with the given (base 0) index in
    the contents of the editable. The value must be less than or equal to the number
    of characters in the editable. A value of -1 indicates that the position should
    be set after the last character of the editable. Note that position is in
    characters, not in bytes.
 */
FALCON_FUNC Editable::set_position( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_editable_set_position( (GtkEditable*)_obj, i_pos->asInteger() );
}


/*#
    @method get_position GtkEditable
    @brief Retrieves the current position of the cursor relative to the start of the content of the editable.
    @return the cursor position

    Note that this position is in characters, not in bytes.
 */
FALCON_FUNC Editable::get_position( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_editable_get_position( (GtkEditable*)_obj ) );
}


/*#
    @method set_editable GtkEditable
    @brief Determines if the user can edit the text in the editable widget or not.
    @param is_editable true if the user is allowed to edit the text in the widget
 */
FALCON_FUNC Editable::set_editable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_editable_set_editable( (GtkEditable*)_obj, i_bool->asInteger() );
}


/*#
    @method get_editable GtkEditable
    @brief Retrieves whether editable is editable.
    @return TRUE if editable is editable.
 */
FALCON_FUNC Editable::get_editable( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_editable_get_editable( (GtkEditable*)_obj ) );
}


} // Gtk
} // Falcon
