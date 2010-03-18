/**
 *  \file gtk_Entry.cpp
 */

#include "gtk_Entry.hpp"

#include "gtk_EntryBuffer.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Entry::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Entry = mod->addClass( "Entry", &Entry::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Widget" ) );
    c_Entry->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
#if GTK_MINOR_VERSION >= 18
    { "get_buffer",             &Entry::get_buffer },
    { "set_buffer",             &Entry::set_buffer },
#endif
    { "set_text",               &Entry::set_text },
    //{ "append_text",            &Entry::foo },
    //{ "prepend_text",           &Entry::foo },
    //{ "set_position",           &Entry::foo },
    { "get_text",               &Entry::get_text },
    { "get_text_length",        &Entry::get_text_length },
    //{ "select_region",          &Entry::foo },
    { "set_visibility",         &Entry::set_visibility },
    { "set_invisible_char",     &Entry::set_invisible_char },
#if GTK_MINOR_VERSION >= 16
    { "unset_invisible_char",   &Entry::unset_invisible_char },
#endif
    //{ "set_editable",           &Entry::foo },
    { "set_max_length",         &Entry::set_max_length },
    { "get_activates_default",  &Entry::get_activates_default },
    //{ "get_has_frame",        &Entry::foo },
    //{ "get_inner_border",        &Entry::foo },
    //{ "get_width_chars",        &Entry::foo },
    //{ "set_activates_default",        &Entry::foo },
    //{ "set_has_frame",        &Entry::foo },
    //{ "set_inner_border",        &Entry::foo },
    //{ "set_width_chars",        &Entry::foo },
    //{ "get_invisible_char",        &Entry::foo },
    //{ "set_alignment",        &Entry::foo },
    //{ "get_alignment",        &Entry::foo },
    //{ "set_overwrite_mode",        &Entry::foo },
    //{ "get_overwrite_mode",        &Entry::foo },
    //{ "get_layout",        &Entry::foo },
    //{ "get_layout_offsets",        &Entry::foo },
    //{ "layout_index_to_text_index",        &Entry::foo },
    //{ "text_index_to_layout_index",        &Entry::foo },
    //{ "get_max_length",        &Entry::foo },
    //{ "get_visibility",        &Entry::foo },
    //{ "set_completion",        &Entry::foo },
    //{ "get_completion",        &Entry::foo },
    //{ "set_cursor_hadjustment",        &Entry::foo },
    //{ "get_cursor_hadjustment",        &Entry::foo },
    //{ "set_progress_fraction",        &Entry::foo },
    //{ "get_progress_fraction",        &Entry::foo },
    //{ "set_progress_pulse_step",        &Entry::foo },
    //{ "get_progress_pulse_step",        &Entry::foo },
    //{ "progress_pulse",        &Entry::foo },
    //{ "set_icon_from_pixbuf",        &Entry::foo },
    //{ "set_icon_from_stock",        &Entry::foo },
    //{ "set_icon_from_icon_name",        &Entry::foo },
    //{ "set_icon_from_gicon",        &Entry::foo },
    //{ "get_icon_storage_type",        &Entry::foo },
    //{ "get_icon_pixbuf",        &Entry::foo },
    //{ "get_icon_stock",        &Entry::foo },
    //{ "get_icon_name",        &Entry::foo },
    //{ "get_icon_gicon",        &Entry::foo },
    //{ "set_icon_activatable",        &Entry::foo },
    //{ "get_icon_activatable",        &Entry::foo },
    //{ "set_icon_sensitive",        &Entry::foo },
    //{ "get_icon_sensitive",        &Entry::foo },
    //{ "get_icon_at_pos",        &Entry::foo },
    //{ "set_icon_tooltip_text",        &Entry::foo },
    //{ "get_icon_tooltip_text",        &Entry::foo },
    //{ "set_icon_tooltip_markup",        &Entry::foo },
    //{ "get_icon_tooltip_markup",        &Entry::foo },
    //{ "set_icon_drag_source",        &Entry::foo },
    //{ "get_current_icon_drag_source",        &Entry::foo },
    //{ "get_icon_window",        &Entry::foo },
    //{ "get_text_window",        &Entry::foo },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Entry, meth->name, meth->cb );
}


/*#
    @class gtk.Entry
    @brief A single line text entry field
    @optparam buffer The buffer (gtk.EntryBuffer) or text (string) to use for the new gtk.Entry.

    The gtk.Entry widget is a single line text entry widget. A fairly large set of
    key bindings are supported by default. If the entered text is longer than the
    allocation of the widget, the widget will scroll so that the cursor position is visible.

    When using an entry for passwords and other sensitive information, it can be
    put into "password mode" using set_visibility(). In this mode, entered
    text is displayed using a 'invisible' character. By default, GTK+ picks the best
    invisible character that is available in the current font, but it can be changed
    with set_invisible_char(). Since 2.16, GTK+ displays a warning when
    Caps Lock or input methods might interfere with entering text in a password entry.
    The warning can be turned off with the "caps-lock-warning" property.

    Since 2.16, gtk.Entry has the ability to display progress or activity information
    behind the text. To make an entry display such information, use
    set_progress_fraction() or set_progress_pulse_step().

    Additionally, gtk.Entry can show icons at either side of the entry. These icons
    can be activatable by clicking, can be set up as drag source and can have tooltips.
    To add an icon, use set_icon_from_gicon() or one of the various other
    functions that set an icon from a stock id, an icon name or a pixbuf. To trigger
    an action when the user clicks an icon, connect to the "icon-press" signal. To allow
    DND operations from an icon, use set_icon_drag_source(). To set a tooltip
    on an icon, use set_icon_tooltip_text() or the corresponding function for markup.

    Note that functionality or information that is only available by clicking on
    an icon in an entry may not be accessible at all to users which are not able
    to use a mouse or other pointing device. It is therefore recommended that any
    such functionality should also be available by other means, e.g. via the context
    menu of the entry.
 */
FALCON_FUNC Entry::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

    Item* i_buf = vm->param( 0 );
    GtkWidget* entry;
    if ( i_buf )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_buf->isNil() ||
            !( i_buf->isString() ||
            i_buf->isOfClass( "EntryBuffer" ) || i_buf->isOfClass( "gtk.EntryBuffer" ) ) )
            throw_inv_params( "[EntryBuffer|S]" );
#endif
        if ( i_buf->isString() )
        {
            AutoCString s( i_buf->asString() );
            entry = gtk_entry_new();
            gtk_entry_set_text( (GtkEntry*) entry, s.c_str() );
        }
#if GTK_MINOR_VERSION >= 18
        else
        {
            GtkEntryBuffer* buf =
                (GtkEntryBuffer*)((GData*)i_buf->asObject()->getUserData())->obj();
            entry = gtk_entry_new_with_buffer( buf );
        }
#endif
    }
    else
        entry = gtk_entry_new();

    Gtk::internal_add_slot( (GObject*) entry );
    self->setUserData( new GData( (GObject*) entry ) );
}


#if GTK_MINOR_VERSION >= 18
/*#
    @method get_buffer gtk.Entry
    @brief Get the gtk.EntryBuffer object which holds the text for this widget.
    @return A gtk.EntryBuffer object.
 */
FALCON_FUNC Entry::get_buffer( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkEntryBuffer* buf = gtk_entry_get_buffer( (GtkEntry*)_obj );
    Item* wki = vm->findWKI( "EntryBuffer" );
    vm->retval( new EntryBuffer( wki->asClass(), buf ) );
}
#endif


#if GTK_MINOR_VERSION >= 18
/*#
    @method set_buffer gtk.Entry
    @brief Set the GtkEntryBuffer object which holds the text for this widget.
    @param buffer (gtk.EntryBuffer)
 */
FALCON_FUNC Entry::set_buffer( VMARG )
{
    Item* i_buf = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_buf || i_buf->isNil() ||
        !( i_buf->isOfClass( "EntryBuffer" ) || i_buf->isOfClass( "gtk.EntryBuffer" ) ) )
        throw_inv_params( "EntryBuffer" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkEntryBuffer* buf = (GtkEntryBuffer*)((GData*)i_buf->asObject()->getUserData())->obj();
    gtk_entry_set_buffer( (GtkEntry*)_obj, buf );
}
#endif


/*#
    @method set_text gtk.Entry
    @brief Sets the text in the widget to the given value, replacing the current contents.
    @param text the new text
 */
FALCON_FUNC Entry::set_text( VMARG )
{
    Item* i_txt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_txt || i_txt->isNil() || !i_txt->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString s( i_txt->asString() );
    gtk_entry_set_text( (GtkEntry*)_obj, s.c_str() );
}


//FALCON_FUNC Entry::append_text( VMARG );

//FALCON_FUNC Entry::prepend_text( VMARG );

//FALCON_FUNC Entry::set_position( VMARG );


/*#
    @method get_text gtk.Entry
    @brief Retrieves the contents of the entry widget.
    @return (string) contents of entry widget.
 */
FALCON_FUNC Entry::get_text( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* txt = gtk_entry_get_text( (GtkEntry*)_obj );
    vm->retval( txt ? new String ( txt ) : new String() );
}


/*#
    @method get_text_length gtk.Entry
    @brief Retrieves the current length of the text in entry.
    @return the current number of characters in gtk.Entry, or 0 if there are none.
 */
FALCON_FUNC Entry::get_text_length( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_entry_get_text_length( (GtkEntry*)_obj ) );
}


//FALCON_FUNC Entry::select_region( VMARG );


/*#
    @method set_visiblity gtk.Entry
    @brief Sets whether the contents of the entry are visible or not.
    @param visible (boolean) true if the contents of the entry are displayed as plaintext

    When visibility is set to false, characters are displayed as the invisible char,
    and will also appear that way when the text in the entry widget is copied elsewhere.

    By default, GTK+ picks the best invisible character available in the current font,
    but it can be changed with set_invisible_char().
 */
FALCON_FUNC Entry::set_visibility( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_visibility( (GtkEntry*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_invisible_char gtk.Entry
    @brief Sets the entry invisible char
    @param ch (integer) a Unicode character

    Sets the character to use in place of the actual text when set_visibility()
    has been called to set text visibility to false. i.e. this is the character
    used in "password mode" to show the user how many characters have been typed.
    By default, GTK+ picks the best invisible char available in the current font.
    If you set the invisible char to 0, then the user will get no feedback at all;
    there will be no text on the screen as they type.
 */
FALCON_FUNC Entry::set_invisible_char( VMARG )
{
    Item* i_chr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chr || i_chr->isNil() || !i_chr->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_invisible_char( (GtkEntry*)_obj, i_chr->asInteger() );
}


#if GTK_MINOR_VERSION >= 16
/*#
    @method unset_invisible_char gtk.Entry
    @brief Unsets the invisible char previously set with set_invisible_char(). So that the default invisible char is used again.
 */
FALCON_FUNC Entry::unset_invisible_char( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_unset_invisible_char( (GtkEntry*)_obj );
}
#endif


//FALCON_FUNC Entry::set_editable( VMARG );


/*#
    @method set_max_length gtk.Entry
    @brief Sets the maximum allowed length of the contents of the widget.
    @param max the maximum length of the entry, or 0 for no maximum. (other than the maximum length of entries.)

    The value passed in will be clamped to the range 0-65536.

    If the current contents are longer than the given length, then they will
    be truncated to fit.
 */
FALCON_FUNC Entry::set_max_length( VMARG )
{
    Item* i_max = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_max || i_max->isNil() || !i_max->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_max_length( (GtkEntry*)_obj, i_max->asInteger() );
}


/*#
    @method get_activates_default gtk.Entry
    @brief Retrieves the value set by set_activates_default().
    @return (boolean) true if the entry will activate the default widget
 */
FALCON_FUNC Entry::get_activates_default( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_entry_get_activates_default( (GtkEntry*)_obj ) );
}


//FALCON_FUNC Entry::get_has_frame( VMARG );

//FALCON_FUNC Entry::get_inner_border( VMARG );

//FALCON_FUNC Entry::get_width_chars( VMARG );

//FALCON_FUNC Entry::set_activates_default( VMARG );

//FALCON_FUNC Entry::set_has_frame( VMARG );

//FALCON_FUNC Entry::set_inner_border( VMARG );

//FALCON_FUNC Entry::set_width_chars( VMARG );

//FALCON_FUNC Entry::get_invisible_char( VMARG );

//FALCON_FUNC Entry::set_alignment( VMARG );

//FALCON_FUNC Entry::get_alignment( VMARG );

//FALCON_FUNC Entry::set_overwrite_mode( VMARG );

//FALCON_FUNC Entry::get_overwrite_mode( VMARG );

//FALCON_FUNC Entry::get_layout( VMARG );

//FALCON_FUNC Entry::get_layout_offsets( VMARG );

//FALCON_FUNC Entry::layout_index_to_text_index( VMARG );

//FALCON_FUNC Entry::text_index_to_layout_index( VMARG );

//FALCON_FUNC Entry::get_max_length( VMARG );

//FALCON_FUNC Entry::get_visibility( VMARG );

//FALCON_FUNC Entry::set_completion( VMARG );

//FALCON_FUNC Entry::get_completion( VMARG );

//FALCON_FUNC Entry::set_cursor_hadjustment( VMARG );

//FALCON_FUNC Entry::get_cursor_hadjustment( VMARG );

//FALCON_FUNC Entry::set_progress_fraction( VMARG );

//FALCON_FUNC Entry::get_progress_fraction( VMARG );

//FALCON_FUNC Entry::set_progress_pulse_step( VMARG );

//FALCON_FUNC Entry::get_progress_pulse_step( VMARG );

//FALCON_FUNC Entry::progress_pulse( VMARG );

//FALCON_FUNC Entry::set_icon_from_pixbuf( VMARG );

//FALCON_FUNC Entry::set_icon_from_stock( VMARG );

//FALCON_FUNC Entry::set_icon_from_icon_name( VMARG );

//FALCON_FUNC Entry::set_icon_from_gicon( VMARG );

//FALCON_FUNC Entry::get_icon_storage_type( VMARG );

//FALCON_FUNC Entry::get_icon_pixbuf( VMARG );

//FALCON_FUNC Entry::get_icon_stock( VMARG );

//FALCON_FUNC Entry::get_icon_name( VMARG );

//FALCON_FUNC Entry::get_icon_gicon( VMARG );

//FALCON_FUNC Entry::set_icon_activatable( VMARG );

//FALCON_FUNC Entry::get_icon_activatable( VMARG );

//FALCON_FUNC Entry::set_icon_sensitive( VMARG );

//FALCON_FUNC Entry::get_icon_sensitive( VMARG );

//FALCON_FUNC Entry::get_icon_at_pos( VMARG );

//FALCON_FUNC Entry::set_icon_tooltip_text( VMARG );

//FALCON_FUNC Entry::get_icon_tooltip_text( VMARG );

//FALCON_FUNC Entry::set_icon_tooltip_markup( VMARG );

//FALCON_FUNC Entry::get_icon_tooltip_markup( VMARG );

//FALCON_FUNC Entry::set_icon_drag_source( VMARG );

//FALCON_FUNC Entry::get_current_icon_drag_source( VMARG );

//FALCON_FUNC Entry::get_icon_window( VMARG );

//FALCON_FUNC Entry::get_text_window( VMARG );


} // Gtk
} // Falcon
