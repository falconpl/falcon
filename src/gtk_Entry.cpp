/**
 *  \file gtk_Entry.cpp
 */

#include "gtk_Entry.hpp"

//#include "gdk_Event.hpp"
#include "gdk_Pixbuf.hpp"
#include "gtk_Adjustment.hpp"
#include "gtk_CellEditable.hpp"
#include "gtk_Editable.hpp"
#include "gtk_EntryBuffer.hpp"
#include "gtk_Menu.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Entry::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Entry = mod->addClass( "GtkEntry", &Entry::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWidget" ) );
    c_Entry->getClassDef()->addInheritance( in );

    c_Entry->setWKS( true );
    c_Entry->getClassDef()->factory( &Entry::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_activate",        &Entry::signal_activate },
    { "signal_backspace",       &Entry::signal_backspace },
    { "signal_copy_clipboard",  &Entry::signal_copy_clipboard },
    { "signal_cut_clipboard",   &Entry::signal_cut_clipboard },
    { "signal_delete_from_cursor",&Entry::signal_delete_from_cursor },
#if 0
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "signal_icon_press",      &Entry::signal_icon_press },
    { "signal_icon_release",    &Entry::signal_icon_release },
#endif
#endif
    { "signal_insert_at_cursor",&Entry::signal_insert_at_cursor },
    { "signal_move_cursor",     &Entry::signal_move_cursor },
    { "signal_paste_clipboard", &Entry::signal_paste_clipboard },
    { "signal_populate_popup",  &Entry::signal_populate_popup },
#if GTK_CHECK_VERSION( 2, 20, 0 )
    { "signal_preedit_changed", &Entry::signal_preedit_changed },
#endif
    { "signal_toggle_overwrite",&Entry::signal_toggle_overwrite },
#if GTK_CHECK_VERSION( 2, 18, 0 )
    { "new_with_buffer",        &Entry::new_with_buffer },
#endif
    { "new_with_max_length",    &Entry::new_with_max_length },
#if GTK_CHECK_VERSION( 2, 18, 0 )
    { "get_buffer",             &Entry::get_buffer },
    { "set_buffer",             &Entry::set_buffer },
#endif
    { "set_text",               &Entry::set_text },
    //{ "append_text",            &Entry::foo },
    //{ "prepend_text",           &Entry::foo },
    //{ "set_position",           &Entry::foo },
    { "get_text",               &Entry::get_text },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_text_length",        &Entry::get_text_length },
#endif
#if 0
    { "select_region",          &Entry::foo },
#endif
    { "set_visibility",         &Entry::set_visibility },
    { "set_invisible_char",     &Entry::set_invisible_char },
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "unset_invisible_char",   &Entry::unset_invisible_char },
#endif
#if 0 // deprecated
    { "set_editable",           &Entry::foo },
#endif
    { "set_max_length",         &Entry::set_max_length },
    { "get_activates_default",  &Entry::get_activates_default },
    { "get_has_frame",          &Entry::get_has_frame },
    //{ "get_inner_border",        &Entry::foo },
    { "get_width_chars",        &Entry::get_width_chars },
    { "set_activates_default",  &Entry::set_activates_default },
    { "set_has_frame",          &Entry::set_has_frame },
    //{ "set_inner_border",        &Entry::set_inner_border },
    { "set_width_chars",        &Entry::set_width_chars },
    { "get_invisible_char",     &Entry::get_invisible_char },
    { "set_alignment",          &Entry::set_alignment },
    { "get_alignment",          &Entry::get_alignment },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "set_overwrite_mode",     &Entry::set_overwrite_mode },
    { "get_overwrite_mode",     &Entry::get_overwrite_mode },
#endif
    //{ "get_layout",        &Entry::foo },
    { "get_layout_offsets",     &Entry::get_layout_offsets },
    { "layout_index_to_text_index",&Entry::layout_index_to_text_index },
    { "text_index_to_layout_index",&Entry::text_index_to_layout_index },
    { "get_max_length",         &Entry::get_max_length },
    { "get_visibility",         &Entry::get_visibility },
    //{ "set_completion",        &Entry::foo },
    //{ "get_completion",        &Entry::foo },
    { "set_cursor_hadjustment", &Entry::set_cursor_hadjustment },
    { "get_cursor_hadjustment", &Entry::get_cursor_hadjustment },
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "set_progress_fraction",  &Entry::set_progress_fraction },
    { "get_progress_fraction",  &Entry::get_progress_fraction },
    { "set_progress_pulse_step",&Entry::set_progress_pulse_step },
    { "get_progress_pulse_step",&Entry::get_progress_pulse_step },
    { "progress_pulse",         &Entry::progress_pulse },
#endif
#if GTK_CHECK_VERSION( 2, 22, 0 )
    //{ "im_context_filter_keypress",     &Entry::im_context_filter_keypress },
    //{ "reset_im_context",     &Entry::foo },
#endif
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "set_icon_from_pixbuf",   &Entry::set_icon_from_pixbuf },
    { "set_icon_from_stock",    &Entry::set_icon_from_stock },
    { "set_icon_from_icon_name",&Entry::set_icon_from_icon_name },
    //{ "set_icon_from_gicon",        &Entry::foo },
    { "get_icon_storage_type",  &Entry::get_icon_storage_type },
    { "get_icon_pixbuf",        &Entry::get_icon_pixbuf },
    { "get_icon_stock",         &Entry::get_icon_stock },
    { "get_icon_name",          &Entry::get_icon_name },
    //{ "get_icon_gicon",        &Entry::foo },
    { "set_icon_activatable",   &Entry::set_icon_activatable },
    { "get_icon_activatable",   &Entry::get_icon_activatable },
    { "set_icon_sensitive",     &Entry::set_icon_sensitive },
    { "get_icon_sensitive",     &Entry::get_icon_sensitive },
    { "get_icon_at_pos",        &Entry::get_icon_at_pos },
    //{ "set_icon_tooltip_text",        &Entry::foo },
    //{ "get_icon_tooltip_text",        &Entry::foo },
    //{ "set_icon_tooltip_markup",        &Entry::foo },
    //{ "get_icon_tooltip_markup",        &Entry::foo },
    //{ "set_icon_drag_source",        &Entry::foo },
    //{ "get_current_icon_drag_source",        &Entry::foo },
#endif
#if GTK_CHECK_VERSION( 2, 20, 0 )
    //{ "get_icon_window",        &Entry::foo },
    //{ "get_text_window",        &Entry::foo },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Entry, meth->name, meth->cb );

    Gtk::Editable::clsInit( mod, c_Entry );
    Gtk::CellEditable::clsInit( mod, c_Entry );
}


Entry::Entry( const Falcon::CoreClass* gen, const GtkEntry* entry )
    :
    Gtk::CoreGObject( gen, (GObject*) entry )
{}


Falcon::CoreObject* Entry::factory( const Falcon::CoreClass* gen, void* entry, bool )
{
    return new Entry( gen, (GtkEntry*) entry );
}


/*#
    @class GtkEntry
    @brief A single line text entry field

    The GtkEntry widget is a single line text entry widget. A fairly large set of
    key bindings are supported by default. If the entered text is longer than the
    allocation of the widget, the widget will scroll so that the cursor position is visible.

    When using an entry for passwords and other sensitive information, it can be
    put into "password mode" using set_visibility(). In this mode, entered
    text is displayed using a 'invisible' character. By default, GTK+ picks the best
    invisible character that is available in the current font, but it can be changed
    with set_invisible_char(). Since 2.16, GTK+ displays a warning when
    Caps Lock or input methods might interfere with entering text in a password entry.
    The warning can be turned off with the "caps-lock-warning" property.

    Since 2.16, GtkEntry has the ability to display progress or activity information
    behind the text. To make an entry display such information, use
    set_progress_fraction() or set_progress_pulse_step().

    Additionally, GtkEntry can show icons at either side of the entry. These icons
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
    if ( self->getObject() )
        return;
    NO_ARGS
    self->setObject( (GObject*) gtk_entry_new() );
}


/*#
    @method signal_activate
    @brief A keybinding signal which gets emitted when the user activates the entry.

    Applications should not connect to it, but may emit it with
    g_signal_emit_by_name() if they need to control activation programmatically.

    The default bindings for this signal are all forms of the Enter key.
 */
FALCON_FUNC Entry::signal_activate( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "activate", (void*) &Entry::on_activate, vm );
}


void Entry::on_activate( GtkEntry* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "activate", "on_activate", (VMachine*)_vm );
}


/*#
    @method signal_backspace
    @brief The backspace signal is a keybinding signal which gets emitted when the user asks for it.

    The default bindings for this signal are Backspace and Shift-Backspace.
 */
FALCON_FUNC Entry::signal_backspace( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "backspace", (void*) &Entry::on_backspace, vm );
}


void Entry::on_backspace( GtkEntry* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "backspace", "on_backspace", (VMachine*)_vm );
}


/*#
    @method signal_copy_clipboard
    @brief The copy-clipboard signal is a keybinding signal which gets emitted to copy the selection to the clipboard.

    The default bindings for this signal are Ctrl-c and Ctrl-Insert.
 */
FALCON_FUNC Entry::signal_copy_clipboard( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "copy_clipboard", (void*) &Entry::on_copy_clipboard, vm );
}


void Entry::on_copy_clipboard( GtkEntry* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "copy_clipboard", "on_copy_clipboard", (VMachine*)_vm );
}


/*#
    @method signal_cut_clipboard
    @brief The cut-clipboard signal is a keybinding signal which gets emitted to cut the selection to the clipboard.

    The default bindings for this signal are Ctrl-x and Shift-Delete.
 */
FALCON_FUNC Entry::signal_cut_clipboard( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "cut_clipboard", (void*) &Entry::on_cut_clipboard, vm );
}


void Entry::on_cut_clipboard( GtkEntry* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "cut_clipboard", "on_cut_clipboard", (VMachine*)_vm );
}


/*#
    @method signal_delete_from_cursor
    @brief The delete-from-cursor signal is a keybinding signal which gets emitted when the user initiates a text deletion.

    If the type is GTK_DELETE_CHARS, GTK+ deletes the selection if there is one,
    otherwise it deletes the requested number of characters.

    The default bindings for this signal are Delete for deleting a character
    and Ctrl-Delete for deleting a word.
 */
FALCON_FUNC Entry::signal_delete_from_cursor( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "delete_from_cursor", (void*) &Entry::on_delete_from_cursor, vm );
}


void Entry::on_delete_from_cursor( GtkEntry* obj, GtkDeleteType type, gint count, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "delete_from_cursor", false );

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
                || !it.asObject()->getMethod( "on_delete_from_cursor", it ) )
            {
                printf(
                "[GtkEntry::on_delete_from_cursor] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) type );
        vm->pushParam( count );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}

#if 0 // todo: missing GdkEvent
#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method signal_icon_press
    @brief The ::icon-press signal is emitted when an activatable icon is clicked.
 */
FALCON_FUNC Entry::signal_icon_press( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "icon_press", (void*) &Entry::on_icon_press, vm );
}


void Entry::on_icon_press( GtkEntry* obj, GtkEntryIconPosition pos, GdkEvent* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "icon_press", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GdkEvent" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_icon_press", it ) )
            {
                printf(
                "[GtkEntry::on_icon_press] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) pos );
        vm->pushParam( new Gdk::Event( wki->asClass(), ev ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_icon_release
    @brief The icon-release signal is emitted on the button release from a mouse click over an activatable icon.
 */
FALCON_FUNC Entry::signal_icon_release( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "icon_release", (void*) &Entry::on_icon_release, vm );
}


void Entry::on_icon_release( GtkEntry* obj, GtkEntryIconPosition pos, GdkEvent* ev, gpointer _vm );
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "icon_release", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GdkEvent" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_icon_release", it ) )
            {
                printf(
                "[GtkEntry::on_icon_release] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) pos );
        vm->pushParam( new Gdk::Event( wki->asClass(), ev ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}
#endif // GTK_CHECK_VERSION( 2, 16, 0 )
#endif

/*#
    @method signal_insert_at_cursor
    @brief The insert-at-cursor signal is a keybinding signal which gets emitted when the user initiates the insertion of a fixed string at the cursor.

    This signal has no default bindings.
 */
FALCON_FUNC Entry::signal_insert_at_cursor( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "insert_at_cursor", (void*) &Entry::on_insert_at_cursor, vm );
}


void Entry::on_insert_at_cursor( GtkEntry* obj, gchar* string, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "insert_at_cursor", false );

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
                || !it.asObject()->getMethod( "on_insert_at_cursor", it ) )
            {
                printf(
                "[GtkEntry::on_insert_at_cursor] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( UTF8String( string ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_move_cursor
    @brief The move-cursor signal is a keybinding signal which gets emitted when the user initiates a cursor movement.

    If the cursor is not visible in entry, this signal causes the viewport to
    be moved instead.

    Applications should not connect to it, but may emit it with
    g_signal_emit_by_name() if they need to control the cursor programmatically.

    The default bindings for this signal come in two variants, the variant with
    the Shift modifier extends the selection, the variant without the Shift
    modifer does not. There are too many key combinations to list them all here.

    - Arrow keys move by individual characters/lines
    - Ctrl-arrow key combinations move by words/paragraphs
    - Home/End keys move to the ends of the buffer

 */
FALCON_FUNC Entry::signal_move_cursor( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "move_cursor", (void*) &Entry::on_move_cursor, vm );
}


void Entry::on_move_cursor( GtkEntry* obj, GtkMovementStep step, gint cnt, gboolean extend, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "move_cursor", false );

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
                || !it.asObject()->getMethod( "on_move_cursor", it ) )
            {
                printf(
                "[GtkEntry::on_move_cursor] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (int64) step );
        vm->pushParam( cnt );
        vm->pushParam( (bool) extend );
        vm->callItem( it, 3 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_paste_clipboard
    @brief The paste-clipboard signal is a keybinding signal which gets emitted to paste the contents of the clipboard into the text view.

    The default bindings for this signal are Ctrl-v and Shift-Insert.
 */
FALCON_FUNC Entry::signal_paste_clipboard( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "paste_clipboard", (void*) &Entry::on_paste_clipboard, vm );
}


void Entry::on_paste_clipboard( GtkEntry* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "paste_clipboard", "on_paste_clipboard", (VMachine*)_vm );
}


/*#
    @method signal_populate_popup
    @brief The populate-popup signal gets emitted before showing the context menu of the entry.

    If you need to add items to the context menu, connect to this signal and
    append your menuitems to the menu.
 */
FALCON_FUNC Entry::signal_populate_popup( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "populate_popup", (void*) &Entry::on_populate_popup, vm );
}


void Entry::on_populate_popup( GtkEntry* obj, GtkMenu* menu, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "populate_popup", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkMenu" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_populate_popup", it ) )
            {
                printf(
                "[GtkEntry::on_populate_popup] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::Menu( wki->asClass(), menu ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method signal_preedit_changed
    @brief If an input method is used, the typed text will not immediately be committed to the buffer. So if you are interested in the text, connect to this signal.
 */
FALCON_FUNC Entry::signal_preedit_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "preedit_changed", (void*) &Entry::on_preedit_changed, vm );
}


void Entry::on_preedit_changed( GtkEntry* obj, gchar* preedit, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "preedit_changed", false );

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
                || !it.asObject()->getMethod( "on_preedit_changed", it ) )
            {
                printf(
                "[GtkEntry::on_preedit_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( UTF8String( preedit ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}
#endif // GTK_CHECK_VERSION( 2, 20, 0 )


/*#
    @method signal_toggle_overwrite
    @brief The toggle-overwrite signal is a keybinding signal which gets emitted to toggle the overwrite mode of the entry.

    The default bindings for this signal is Insert.
 */
FALCON_FUNC Entry::signal_toggle_overwrite( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "toggle_overwrite", (void*) &Entry::on_toggle_overwrite, vm );
}


void Entry::on_toggle_overwrite( GtkEntry* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "toggle_overwrite", "on_toggle_overwrite", (VMachine*)_vm );
}


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method new_with_buffer
    @brief Creates a new entry with the specified text buffer.
    @param buffer The GtkEntryBuffer to use for the new GtkEntry.
 */
FALCON_FUNC Entry::new_with_buffer( VMARG )
{
    Item* i_buf = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_buf || !i_buf->isObject() || !IS_DERIVED( i_buf, GtkEntryBuffer ) )
        throw_inv_params( "GtkEntryBuffer" );
#endif
    GtkEntryBuffer* buf = (GtkEntryBuffer*) COREGOBJECT( i_buf )->getObject();
    GtkWidget* entry = gtk_entry_new_with_buffer( buf );
    vm->retval( new Gtk::Entry( vm->findWKI( "GtkEntry" )->asClass(),
                                (GtkEntry*) entry ) );
}
#endif


/*#
    @method new_with_max_length
    @brief Creates a new GtkEntry widget with the given maximum length.
    @param max the maximum length of the entry, or 0 for no maximum. (other than the maximum length of entries.) The value passed in will be clamped to the range 0-65536.

    gtk_entry_new_with_max_length has been deprecated since version 2.0 and
    should not be used in newly-written code. Use gtk_entry_set_max_length() instead.
 */
FALCON_FUNC Entry::new_with_max_length( VMARG )
{
    Item* i_max = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_max || !i_max->isInteger() )
        throw_inv_params( "I" );
#endif
    GtkWidget* entry = gtk_entry_new_with_max_length( i_max->asInteger() );
    vm->retval( new Gtk::Entry( vm->findWKI( "GtkEntry" )->asClass(),
                                (GtkEntry*) entry ) );
}


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method get_buffer GtkEntry
    @brief Get the GtkEntryBuffer object which holds the text for this widget.
    @return A GtkEntryBuffer object.
 */
FALCON_FUNC Entry::get_buffer( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkEntryBuffer* buf = gtk_entry_get_buffer( (GtkEntry*)_obj );
    vm->retval( new Gtk::EntryBuffer( vm->findWKI( "GtkEntryBuffer" )->asClass(), buf ) );
}
#endif


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method set_buffer GtkEntry
    @brief Set the GtkEntryBuffer object which holds the text for this widget.
    @param buffer a GtkEntryBuffer
 */
FALCON_FUNC Entry::set_buffer( VMARG )
{
    Item* i_buf = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_buf || i_buf->isObject() || !IS_DERIVED( i_buf, GtkEntryBuffer ) )
        throw_inv_params( "GtkEntryBuffer" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkEntryBuffer* buf = (GtkEntryBuffer*) COREGOBJECT( i_buf )->getObject();
    gtk_entry_set_buffer( (GtkEntry*)_obj, buf );
}
#endif


/*#
    @method set_text GtkEntry
    @brief Sets the text in the widget to the given value, replacing the current contents.
    @param text the new text
 */
FALCON_FUNC Entry::set_text( VMARG )
{
    Item* i_txt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_txt || !i_txt->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString txt( i_txt->asString() );
    gtk_entry_set_text( (GtkEntry*)_obj, txt.c_str() );
}


//FALCON_FUNC Entry::append_text( VMARG );

//FALCON_FUNC Entry::prepend_text( VMARG );

//FALCON_FUNC Entry::set_position( VMARG );


/*#
    @method get_text GtkEntry
    @brief Retrieves the contents of the entry widget.
    @return the contents of the widget as a string.
 */
FALCON_FUNC Entry::get_text( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* txt = gtk_entry_get_text( (GtkEntry*)_obj );
    if ( txt )
        vm->retval( UTF8String( txt ) );
    else
        vm->retval( UTF8String( "" ) );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_text_length GtkEntry
    @brief Retrieves the current length of the text in entry.
    @return the current number of characters in GtkEntry, or 0 if there are none.
 */
FALCON_FUNC Entry::get_text_length( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_entry_get_text_length( (GtkEntry*)_obj ) );
}
#endif


#if 0 // deprecated
FALCON_FUNC Entry::select_region( VMARG );
#endif


/*#
    @method set_visiblity GtkEntry
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
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_visibility( (GtkEntry*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_invisible_char GtkEntry
    @brief Sets the entry invisible char
    @param ch a character

    Sets the character to use in place of the actual text when set_visibility()
    has been called to set text visibility to false. i.e. this is the character
    used in "password mode" to show the user how many characters have been typed.
    By default, GTK+ picks the best invisible char available in the current font.
    If you set the invisible char to "", then the user will get no feedback at all;
    there will be no text on the screen as they type.
 */
FALCON_FUNC Entry::set_invisible_char( VMARG )
{
    Item* i_chr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_chr || !i_chr->isString() )
        throw_inv_params( "S" );
#endif
    String* chr = i_chr->asString();
    uint32 uni = chr->length() ? chr->getCharAt( 0 ) : 0;
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_invisible_char( (GtkEntry*)_obj, uni );
}


#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method unset_invisible_char GtkEntry
    @brief Unsets the invisible char previously set with set_invisible_char().

    So that the default invisible char is used again.
 */
FALCON_FUNC Entry::unset_invisible_char( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_entry_unset_invisible_char( (GtkEntry*)_obj );
}
#endif


#if 0 // deprecated
FALCON_FUNC Entry::set_editable( VMARG );
#endif


/*#
    @method set_max_length GtkEntry
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
    if ( !i_max || !i_max->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_max_length( (GtkEntry*)_obj, i_max->asInteger() );
}


/*#
    @method get_activates_default GtkEntry
    @brief Retrieves the value set by set_activates_default().
    @return (boolean) true if the entry will activate the default widget
 */
FALCON_FUNC Entry::get_activates_default( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_entry_get_activates_default( (GtkEntry*)_obj ) );
}


/*#
    @method get_has_frame GtkEntry
    @brief Gets the value set by gtk_entry_set_has_frame().
    @return whether the entry has a beveled frame
 */
FALCON_FUNC Entry::get_has_frame( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_entry_get_has_frame( (GtkEntry*)_obj ) );
}


//FALCON_FUNC Entry::get_inner_border( VMARG );


/*#
    @method get_width_chars GtkEntry
    @brief Gets the value set by gtk_entry_set_width_chars().
    @return number of chars to request space for, or negative if unset
 */
FALCON_FUNC Entry::get_width_chars( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_entry_get_width_chars( (GtkEntry*)_obj ) );
}


/*#
    @method set_activates_default GtkEntry
    @brief If setting is TRUE, pressing Enter in the entry will activate the default widget for the window containing the entry.
    @param setting TRUE to activate window's default widget on Enter keypress

    This usually means that the dialog box containing the entry will be closed, since the default widget is usually one of the dialog buttons.

    (For experts: if setting is TRUE, the entry calls gtk_window_activate_default()
    on the window containing the entry, in the default handler for the "activate" signal.)
 */
FALCON_FUNC Entry::set_activates_default( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_activates_default( (GtkEntry*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_has_frame GtkEntry
    @brief Sets whether the entry has a beveled frame around it.
    @param setting new value (boolean).
 */
FALCON_FUNC Entry::set_has_frame( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_has_frame( (GtkEntry*)_obj, (gboolean) i_bool->asBoolean() );
}


//FALCON_FUNC Entry::set_inner_border( VMARG );


/*#
    @method set_width_chars GtkEntry
    @brief Changes the size request of the entry to be about the right size for n_chars characters.
    @param n_chars width in chars

    Note that it changes the size request, the size can still be affected by how
    you pack the widget into containers. If n_chars is -1, the size reverts to
    the default entry size.
 */
FALCON_FUNC Entry::set_width_chars( VMARG )
{
    Item* i_n = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_n || !i_n->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_width_chars( (GtkEntry*)_obj, i_n->asInteger() );
}


/*#
    @method get_invisible_char GtkEntry
    @brief Retrieves the character displayed in place of the real characters for entries with visibility set to false.
    @return the current invisible char, or "", if the entry does not show invisible text at all.
 */
FALCON_FUNC Entry::get_invisible_char( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gunichar uni = gtk_entry_get_invisible_char( (GtkEntry*)_obj );
    if ( uni )
    {
        String* chr = new String( 1 );
        chr->setCharAt( 0, uni );
        vm->retval( chr->bufferize() );
    }
    else
        vm->retval( UTF8String( "" ) );
}


/*#
    @method set_alignment
    @brief Sets the alignment for the contents of the entry.
    @param xalign The horizontal alignment, from 0 (left) to 1 (right). Reversed for RTL layouts

    This controls the horizontal positioning of the contents when the displayed
    text is shorter than the width of the entry.
 */
FALCON_FUNC Entry::set_alignment( VMARG )
{
    Item* i_x = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isOrdinal() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_alignment( (GtkEntry*)_obj, i_x->forceNumeric() );
}


/*#
    @method get_alignment
    @brief Gets the value set by gtk_entry_set_alignment().
    @return the alignment
 */
FALCON_FUNC Entry::get_alignment( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (numeric) gtk_entry_get_alignment( (GtkEntry*)_obj ) );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method set_overwrite_mode
    @brief Sets whether the text is overwritten when typing in the GtkEntry.
    @param overwrite new value (boolean).
 */
FALCON_FUNC Entry::set_overwrite_mode( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_overwrite_mode( (GtkEntry*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_overwrite_mode
    @brief Gets the value set by gtk_entry_set_overwrite_mode().
    @return whether the text is overwritten when typing.
 */
FALCON_FUNC Entry::get_overwrite_mode( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_entry_get_overwrite_mode( (GtkEntry*)_obj ) );
}
#endif // GTK_CHECK_VERSION( 2, 14, 0 )


//FALCON_FUNC Entry::get_layout( VMARG );


/*#
    @method get_layout_offsets
    @brief Obtains the position of the PangoLayout used to render text in the entry, in widget coordinates.
    @return an array ( X offset, Y offset ).

    Useful if you want to line up the text in an entry with some other text,
    e.g. when using the entry to implement editable cells in a sheet widget.

    Also useful to convert mouse events into coordinates inside the PangoLayout,
    e.g. to take some action if some part of the entry text is clicked.

    Note that as the user scrolls around in the entry the offsets will change;
    you'll need to connect to the "notify::scroll-offset" signal to track this.
    Remember when using the PangoLayout functions you need to convert to and
    from pixels using PANGO_PIXELS() or PANGO_SCALE.

    Keep in mind that the layout text may contain a preedit string, so
    gtk_entry_layout_index_to_text_index() and gtk_entry_text_index_to_layout_index()
    are needed to convert byte indices in the layout to byte indices in the
    entry contents.
 */
FALCON_FUNC Entry::get_layout_offsets( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint x, y;
    gtk_entry_get_layout_offsets( (GtkEntry*)_obj, &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( x );
    arr->append( y );
    vm->retval( arr );
}


/*#
    @method layout_index_to_text_index
    @brief Converts from a position in the entry contents (returned by gtk_entry_get_text()) to a position in the entry's PangoLayout (returned by gtk_entry_get_layout(), with text retrieved via pango_layout_get_text()).
    @param layout_index byte index into the entry layout text
    @return byte index into the entry contents
 */
FALCON_FUNC Entry::layout_index_to_text_index( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_entry_layout_index_to_text_index( (GtkEntry*)_obj,
                                                      i_idx->asInteger() ) );
}


/*#
    @method text_index_to_layout_index
    @brief Converts from a position in the entry's PangoLayout (returned by gtk_entry_get_layout()) to a position in the entry contents (returned by gtk_entry_get_text()).
    @param text_index byte index into the entry contents
    @return byte index into the entry layout text
 */
FALCON_FUNC Entry::text_index_to_layout_index( VMARG )
{
    Item* i_idx = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_idx || !i_idx->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_entry_text_index_to_layout_index( (GtkEntry*)_obj,
                                                      i_idx->asInteger() ) );
}


/*#
    @method get_max_length
    @brief Retrieves the maximum allowed length of the text in entry.
    @return the maximum allowed number of characters in GtkEntry, or 0 if there is no maximum.
 */
FALCON_FUNC Entry::get_max_length( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_entry_get_max_length( (GtkEntry*)_obj ) );
}


/*#
    @method get_visibility
    @brief Retrieves whether the text in entry is visible.
    @return TRUE if the text is currently visible
 */
FALCON_FUNC Entry::get_visibility( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_entry_get_visibility( (GtkEntry*)_obj ) );
}


//FALCON_FUNC Entry::set_completion( VMARG );

//FALCON_FUNC Entry::get_completion( VMARG );


/*#
    @method set_cursor_hadjustment
    @brief Hooks up an adjustment to the cursor position in an entry, so that when the cursor is moved, the adjustment is scrolled to show that position.
    @param adjustment an adjustment which should be adjusted when the cursor is moved, or NULL

    See gtk_scrolled_window_get_hadjustment() for a typical way of obtaining
    the adjustment.

    The adjustment has to be in pixel units and in the same coordinate system
    as the entry.
 */
FALCON_FUNC Entry::set_cursor_hadjustment( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) ) )
        throw_inv_params( "[GtkAdjustment]" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL
                    : (GtkAdjustment*) COREGOBJECT( i_adj )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_cursor_hadjustment( (GtkEntry*)_obj, adj );
}


/*#
    @method
    @brief Retrieves the horizontal cursor adjustment for the entry
    @return the horizontal cursor adjustment, or NULL  if none has been set.
 */
FALCON_FUNC Entry::get_cursor_hadjustment( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkAdjustment* adj = gtk_entry_get_cursor_hadjustment( (GtkEntry*)_obj );
    vm->retval( new Gtk::Adjustment( vm->findWKI( "GtkAdjustment" )->asClass(), adj ) );
}


#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method set_progress_fraction
    @brief Causes the entry's progress indicator to "fill in" the given fraction of the bar.
    @param fraction fraction of the task that's been completed.

    The fraction should be between 0.0 and 1.0, inclusive.
 */
FALCON_FUNC Entry::set_progress_fraction( VMARG )
{
    Item* i_fr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_fr || !i_fr->isOrdinal() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_progress_fraction( (GtkEntry*)_obj, i_fr->forceNumeric() );
}


/*#
    @method get_progress_fraction
    @brief Returns the current fraction of the task that's been completed.
    @return a fraction from 0.0 to 1.0
 */
FALCON_FUNC Entry::get_progress_fraction( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (numeric) gtk_entry_get_progress_fraction( (GtkEntry*)_obj ) );
}


/*#
    @method set_progress_pulse_step
    @brief Sets the fraction of total entry width to move the progress bouncing block for each call to gtk_entry_progress_pulse().
    @param fraction fraction between 0.0 and 1.0
 */
FALCON_FUNC Entry::set_progress_pulse_step( VMARG )
{
    Item* i_fr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_fr || !i_fr->isOrdinal() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_progress_pulse_step( (GtkEntry*)_obj, i_fr->forceNumeric() );
}


/*#
    @method get_progress_pulse_step
    @brief Retrieves the pulse step set with gtk_entry_set_progress_pulse_step().
    @return a fraction from 0.0 to 1.0
 */
FALCON_FUNC Entry::get_progress_pulse_step( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (numeric) gtk_entry_get_progress_pulse_step( (GtkEntry*)_obj ) );
}


/*#
    @method progress_pulse
    @brief Indicates that some progress is made, but you don't know how much.

    Causes the entry's progress indicator to enter "activity mode," where a block
    bounces back and forth. Each call to gtk_entry_progress_pulse() causes the
    block to move by a little bit (the amount of movement per pulse is determined
    by gtk_entry_set_progress_pulse_step()).
 */
FALCON_FUNC Entry::progress_pulse( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_entry_progress_pulse( (GtkEntry*)_obj );
}
#endif // GTK_CHECK_VERSION( 2, 16, 0 )


#if GTK_CHECK_VERSION( 2, 22, 0 )
//FALCON_FUNC Entry::im_context_filter_keypress( VMARG )
//FALCON_FUNC Entry::reset_im_context
#endif


#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method set_icon_from_pixbuf
    @brief Sets the icon shown in the specified position using a pixbuf.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @param pixbuf A GdkPixbuf, or NULL.

    If pixbuf is NULL, no icon will be shown in the specified position.
 */
FALCON_FUNC Entry::set_icon_from_pixbuf( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_pix = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_pix || !( i_pix->isNil() || ( i_pix->isObject()
        && IS_DERIVED( i_pix, GdkPixbuf ) ) ) )
        throw_inv_params( "GtkEntryIconPosition,[GdkPixbuf]" );
#endif
    GdkPixbuf* pix = i_pix->isNil() ? NULL
                    : (GdkPixbuf*) COREGOBJECT( i_pix )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_icon_from_pixbuf( (GtkEntry*)_obj,
                                    (GtkEntryIconPosition) i_pos->asInteger(),
                                    pix );
}


/*#
    @method set_icon_from_stock
    @brief Sets the icon shown in the entry at the specified position from a stock image.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @param stock_id The name of the stock item, or NULL.

    If stock_id is NULL, no icon will be shown in the specified position.
 */
FALCON_FUNC Entry::set_icon_from_stock( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_stock = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_stock || !( i_stock->isNil() || i_stock->isString() ) )
        throw_inv_params( "GtkEntryIconPosition,[S]" );
#endif
    MYSELF;
    GET_OBJ( self );
    if ( i_stock->isString() )
    {
        AutoCString stock( i_stock->asString() );
        gtk_entry_set_icon_from_stock( (GtkEntry*)_obj,
                                       (GtkEntryIconPosition) i_pos->asInteger(),
                                       stock.c_str() );
    }
    else
        gtk_entry_set_icon_from_stock( (GtkEntry*)_obj,
                                       (GtkEntryIconPosition) i_pos->asInteger(),
                                       NULL );
}


/*#
    @method set_icon_from_icon_name
    @brief Sets the icon shown in the entry at the specified position from the current icon theme.
    @param icon_pos The position at which to set the icon (GtkEntryIconPosition).
    @param icon_name An icon name, or NULL.

    If the icon name isn't known, a "broken image" icon will be displayed instead.

    If icon_name is NULL, no icon will be shown in the specified position.
 */
FALCON_FUNC Entry::set_icon_from_icon_name( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_nm = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_nm || !( i_nm->isNil() || i_nm->isString() ) )
        throw_inv_params( "GtkEntryIconPosition,[S]" );
#endif
    MYSELF;
    GET_OBJ( self );
    if ( i_nm->isString() )
    {
        AutoCString nm( i_nm->asString() );
        gtk_entry_set_icon_from_icon_name( (GtkEntry*)_obj,
                                           (GtkEntryIconPosition) i_pos->asInteger(),
                                           nm.c_str() );
    }
    else
        gtk_entry_set_icon_from_icon_name( (GtkEntry*)_obj,
                                           (GtkEntryIconPosition) i_pos->asInteger(),
                                           NULL );
}


//FALCON_FUNC Entry::set_icon_from_gicon( VMARG );


/*#
    @method get_icon_storage_type
    @brief Gets the type of representation being used by the icon to store image data.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return image representation being used

    If the icon has no image data, the return value will be GTK_IMAGE_EMPTY.
 */
FALCON_FUNC Entry::get_icon_storage_type( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_entry_get_icon_storage_type( (GtkEntry*)_obj,
                                (GtkEntryIconPosition) i_pos->asInteger() ) );
}


/*#
    @method get_icon_pixbuf
    @brief Retrieves the image used for the icon.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return A GdkPixbuf, or NULL if no icon is set for this position.

    Unlike the other methods of setting and getting icon data, this method will
    work regardless of whether the icon was set using a GdkPixbuf, a GIcon,
    a stock item, or an icon name.
 */
FALCON_FUNC Entry::get_icon_pixbuf( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    GdkPixbuf* pix = gtk_entry_get_icon_pixbuf( (GtkEntry*)_obj,
                                    (GtkEntryIconPosition) i_pos->asInteger() );
    vm->retval( new Gdk::Pixbuf( vm->findWKI( "GdkPixbuf" )->asClass(), pix ) );
}


/*#
    @method get_icon_stock
    @brief Retrieves the stock id used for the icon, or NULL if there is no icon or if the icon was set by some other method (e.g., by pixbuf, icon name or gicon).
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return A stock id, or NULL if no icon is set or if the icon wasn't set from a stock id.
 */
FALCON_FUNC Entry::get_icon_stock( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* stock = gtk_entry_get_icon_stock( (GtkEntry*)_obj,
                                    (GtkEntryIconPosition) i_pos->asInteger() );
    if ( stock )
        vm->retval( UTF8String( stock ) );
    else
        vm->retnil();
}


/*#
    @method get_icon_name
    @brief Retrieves the icon name used for the icon, or NULL if there is no icon or if the icon was set by some other method (e.g., by pixbuf, stock or gicon).
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return An icon name, or NULL if no icon is set or if the icon wasn't set from an icon name
 */
FALCON_FUNC Entry::get_icon_name( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* nm = gtk_entry_get_icon_name( (GtkEntry*)_obj,
                                    (GtkEntryIconPosition) i_pos->asInteger() );
    if ( nm )
        vm->retval( UTF8String( nm ) );
    else
        vm->retnil();
}


//FALCON_FUNC Entry::get_icon_gicon( VMARG );


/*#
    @method set_icon_activatable
    @brief Sets whether the icon is activatable.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @param activatable TRUE if the icon should be activatable.
 */
FALCON_FUNC Entry::set_icon_activatable( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_bool = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "GtkEntryIconPosition,B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_icon_activatable( (GtkEntry*)_obj,
                                    (GtkEntryIconPosition) i_pos->asInteger(),
                                    (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_icon_activatable
    @brief Returns whether the icon is activatable.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return TRUE if the icon is activatable.
 */
FALCON_FUNC Entry::get_icon_activatable( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_entry_get_icon_activatable( (GtkEntry*)_obj,
                                (GtkEntryIconPosition) i_pos->asInteger() ) );
}


/*#
    @method set_icon_sensitive
    @brief Sets the sensitivity for the specified icon.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @param sensitive Specifies whether the icon should appear sensitive or insensitive
 */
FALCON_FUNC Entry::set_icon_sensitive( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_bool = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "GtkEntryIconPosition,B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_entry_set_icon_sensitive( (GtkEntry*)_obj,
                                  (GtkEntryIconPosition) i_pos->asInteger(),
                                  (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_icon_sensitive
    @brief Returns whether the icon appears sensitive or insensitive.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return TRUE if the icon is sensitive.
 */
FALCON_FUNC Entry::get_icon_sensitive( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_entry_get_icon_sensitive( (GtkEntry*)_obj,
                                (GtkEntryIconPosition) i_pos->asInteger() ) );
}


/*#
    @method get_icon_at_pos
    @brief Finds the icon at the given position and return its index.
    @param x the x coordinate of the position to find
    @param y the y coordinate of the position to find
    @return the index of the icon at the given position, or -1.

    If x, y doesn't lie inside an icon, -1 is returned. This function is
    intended for use in a "query-tooltip" signal handler.
 */
FALCON_FUNC Entry::get_icon_at_pos( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_entry_get_icon_at_pos( (GtkEntry*)_obj,
                                           i_x->asInteger(), i_y->asInteger() ) );
}


/*#
    @method set_icon_tooltip_text
    @brief Sets tooltip as the contents of the tooltip for the icon at the specified position.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @param tooltip the contents of the tooltip for the icon, or NULL.

    Use NULL for tooltip to remove an existing tooltip.
 */
FALCON_FUNC Entry::set_icon_tooltip_text( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_tip = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger()
        || !i_tip || !( i_tip->isNil() || i_tip->isString() ) )
        throw_inv_params( "GtkEntryIconPosition,[S]" );
#endif
    MYSELF;
    GET_OBJ( self );
    if ( i_tip->isString() )
    {
        AutoCString tip( i_tip->asString() );
        gtk_entry_set_icon_tooltip_text( (GtkEntry*)_obj,
                                         (GtkEntryIconPosition) i_pos->asInteger(),
                                         tip.c_str() );
    }
    else
        gtk_entry_set_icon_tooltip_text( (GtkEntry*)_obj,
                                         (GtkEntryIconPosition) i_pos->asInteger(),
                                         NULL );
}


/*#
    @method get_icon_tooltip_text
    @brief Gets the contents of the tooltip on the icon at the specified position in entry.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return the tooltip text, or NULL.
 */
FALCON_FUNC Entry::get_icon_tooltip_text( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    gchar* txt = gtk_entry_get_icon_tooltip_text( (GtkEntry*)_obj,
                                    (GtkEntryIconPosition) i_pos->asInteger() );
    if ( txt )
    {
        vm->retval( UTF8String( txt ) );
        g_free( txt );
    }
    else
        vm->retnil();
}


/*#
    @method set_icon_tooltip_markup
    @brief Sets tooltip as the contents of the tooltip for the icon at the specified position.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @param tooltip the contents of the tooltip for the icon, or NULL.

    tooltip is assumed to be marked up with the Pango text markup language.

    Use NULL for tooltip to remove an existing tooltip.

    See also gtk_widget_set_tooltip_markup() and gtk_enty_set_icon_tooltip_text().
 */
FALCON_FUNC Entry::set_icon_tooltip_markup( VMARG )
{
    Item* i_pos = vm->param( 0 );
    Item* i_tip = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->asInteger()
        || !i_tip || !( i_tip->isNil() || i_tip->isString() ) )
        throw_inv_params( "GtkEntryIconPosition,[S]" );
#endif
    MYSELF;
    GET_OBJ( self );
    if ( i_tip->isString() )
    {
        AutoCString tip( i_tip->asString() );
        gtk_entry_set_icon_tooltip_markup( (GtkEntry*)_obj,
                                           (GtkEntryIconPosition) i_pos->asInteger(),
                                           tip.c_str() );
    }
    else
        gtk_entry_set_icon_tooltip_markup( (GtkEntry*)_obj,
                                           (GtkEntryIconPosition) i_pos->asInteger(),
                                           NULL );
}


/*#
    @method get_icon_tooltip_markup
    @brief Gets the contents of the tooltip on the icon at the specified position in entry.
    @param icon_pos Icon position (GtkEntryIconPosition).
    @return the tooltip text, or NULL.
 */
FALCON_FUNC Entry::get_icon_tooltip_markup( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkEntryIconPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    gchar* txt = gtk_entry_get_icon_tooltip_markup( (GtkEntry*)_obj,
                                    (GtkEntryIconPosition) i_pos->asInteger() );
    if ( txt )
    {
        vm->retval( UTF8String( txt ) );
        g_free( txt );
    }
    else
        vm->retnil();
}



//FALCON_FUNC Entry::set_icon_drag_source( VMARG );

//FALCON_FUNC Entry::get_current_icon_drag_source( VMARG );
#endif // GTK_CHECK_VERSION( 2, 16, 0 )

#if GTK_CHECK_VERSION( 2, 20, 0 )
//FALCON_FUNC Entry::get_icon_window( VMARG );

//FALCON_FUNC Entry::get_text_window( VMARG );
#endif


} // Gtk
} // Falcon
