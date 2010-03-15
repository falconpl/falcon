/**
 *  \file gtk_Window.cpp
 */

#include "gtk_Window.hpp"

#include "gtk_Widget.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Window::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Window = mod->addClass( "Window", &Window::init )
        ->addParam( "type" );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Bin" ) );
    c_Window->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    //{ "signal_activate_default"      &Window::set_resizable },
    //{ "signal_activate_focus"      &Window::set_resizable },
    //{ "signal_frame_event"      &Window::set_resizable },
    //{ "signal_keys_changed"      &Window::set_resizable },
    //{ "signal_set_focus"          &Window::set_resizable },
    { "set_title",          &Window::set_title },
    { "set_resizable",      &Window::set_resizable },
    { "get_resizable",      &Window::get_resizable },
    { "activate_focus",     &Window::activate_focus },
    { "activate_default",   &Window::activate_default },
    { "set_modal",          &Window::set_modal },
    { "set_default_size",   &Window::set_default_size },
    //{ "set_geomerty_hints", &Window::set_geometry_hints },
    { "set_gravity",        &Window::set_gravity },
    { "get_gravity",        &Window::get_gravity },
    { "set_position",       &Window::set_position },
    { "set_transient_for",  &Window::set_transient_for },
    { "set_destroy_with_parent",&Window::set_destroy_with_parent },
    //{ "set_screen",        &Window::set_gravity },
    //{ "get_screen",        &Window::set_gravity },
    { "is_active",          &Window::is_active },
    { "has_toplevel_focus", &Window::has_toplevel_focus },
    //{ "list_toplevels",        &Window::set_gravity },
    { "add_mnemonic",       &Window::add_mnemonic },
    { "remove_mnemonic",    &Window::remove_mnemonic },
    { "mnemonic_activate",        &Window::set_gravity },
    //{ "activate_key",        &Window::set_gravity },
    //{ "propagate_key_event",        &Window::set_gravity },
    { "get_focus",          &Window::get_focus },
    { "set_focus",          &Window::set_focus },
    { "get_default_widget", &Window::get_default_widget },
    { "set_default",        &Window::set_default },
    { "present",            &Window::present },
    { "present_with_time",  &Window::present_with_time },
    { "iconify",            &Window::iconify },
    { "deiconify",          &Window::deiconify },
    { "stick",              &Window::stick },
    { "unstick",            &Window::unstick },
    { "maximize",           &Window::maximize },
    { "unmaximize",         &Window::unmaximize },
    { "fullscreen",         &Window::fullscreen },
    { "unfullscreen",       &Window::unfullscreen },
    { "set_keep_above",     &Window::set_keep_above },
    { "set_keep_below",     &Window::set_keep_below },
    { "begin_resize_drag",  &Window::begin_resize_drag },
    { "begin_move_drag",    &Window::begin_move_drag },
    { "set_decorated",      &Window::set_decorated },
    { "set_deletable",      &Window::set_deletable },
    { "set_frame_dimensions",&Window::set_frame_dimensions },
    { "set_has_frame",      &Window::set_has_frame },
    //{ "set_mnemonic_modifier",        &Window::set_gravity },
    //{ "set_type_hint",        &Window::set_gravity },
    //{ "set_skip_taskbar_hint",        &Window::set_gravity },
    //{ "set_skip_pager_hint",        &Window::set_gravity },
    //{ "set_urgency_hint",        &Window::set_gravity },
    //{ "set_accept_focus",        &Window::set_gravity },
    //{ "set_focus_on_map",        &Window::set_gravity },
    //{ "set_startup_id",        &Window::set_gravity },
    //{ "set_role",        &Window::set_gravity },
    //{ "get_decorated",        &Window::set_gravity },
    //{ "get_deletable",        &Window::set_gravity },
    //{ "get_default_icon_list",        &Window::set_gravity },
    //{ "get_default_icon_name",        &Window::set_gravity },
    //{ "get_default_size",        &Window::set_gravity },
    //{ "get_destroy_with_parent",        &Window::set_gravity },
    //{ "get_frame_dimensions",        &Window::set_gravity },
    //{ "get_has_frame",        &Window::set_gravity },
    //{ "get_icon",        &Window::set_gravity },
    //{ "get_icon_list",        &Window::set_gravity },
    //{ "get_icon_name",        &Window::set_gravity },
    //{ "get_mnemonic_modifier",        &Window::set_gravity },
    //{ "get_modal",        &Window::set_gravity },
    //{ "get_position",        &Window::set_gravity },
    //{ "get_role",        &Window::set_gravity },
    //{ "get_size",        &Window::set_gravity },

    { "get_title",          &Window::get_title },

    //{ "get_transient_for",        &Window::set_gravity },
    //{ "get_type_hint",        &Window::set_gravity },
    //{ "get_skip_taskbar_hint",        &Window::set_gravity },
    //{ "get_skip_pager_hint",        &Window::set_gravity },
    //{ "get_urgency_hint",        &Window::set_gravity },
    //{ "get_accept_focus",        &Window::set_gravity },
    //{ "get_focus_on_map",        &Window::set_gravity },
    //{ "get_group",        &Window::set_gravity },
    //{ "get_window_type",        &Window::set_gravity },
    //{ "move",        &Window::set_gravity },
    //{ "parse_geometry",        &Window::set_gravity },
    //{ "reshow_with_initial_size",        &Window::set_gravity },
    //{ "resize",        &Window::set_gravity },
    //{ "set_default_icon_list",        &Window::set_gravity },
    //{ "set_default_icon",        &Window::set_gravity },
    //{ "set_default_icon_from_file",        &Window::set_gravity },
    //{ "set_default_icon_name",        &Window::set_gravity },
    //{ "set_icon",        &Window::set_gravity },
    //{ "set_icon_list",        &Window::set_gravity },
    //{ "set_icon_from_file",        &Window::set_gravity },
    //{ "set_icon_name",        &Window::set_gravity },
    //{ "set_auto_startup_notification",        &Window::set_gravity },
    //{ "get_opacity",        &Window::set_gravity },
    //{ "set_opacity",        &Window::set_gravity },
    //{ "get_mnemonics_visible",        &Window::set_gravity },
    //{ "set_mnemonics_visible",        &Window::set_gravity },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Window, meth->name, meth->cb );
}

/*#
    @class gtk.Window
    @optparam type gtk.WINDOW_TOPLEVEL (default) or gtk.WINDOW_POPUP
    @raise ParamError Invalid window type

    @prop title Window title
 */

/*#
    @init gtk.Window
 */
FALCON_FUNC Window::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
        return;

    Item* i_wtype = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_wtype && ( i_wtype->isNil() || !i_wtype->isInteger() ) )
    {
        throw_inv_params( "I" );
    }
#endif
    GtkWindowType gwt = GTK_WINDOW_TOPLEVEL;

    if ( i_wtype )
    {
        const int wtype = i_wtype->asInteger();

        switch ( wtype )
        {
        case GTK_WINDOW_TOPLEVEL:
            break;
        case GTK_WINDOW_POPUP:
            gwt = GTK_WINDOW_POPUP;
            break;
        default:
            throw_inv_params( FAL_STR( gtk_e_inv_window_type_ ) );
        }
    }

    GtkWidget* win = gtk_window_new( gwt );
    Gtk::internal_add_slot( (GObject*) win );
    self->setUserData( new GData( (GObject*) win ) );
}


//FALCON_FUNC Window::signal_activate_default( VMARG );

//FALCON_FUNC Window::signal_activate_focus( VMARG );

//FALCON_FUNC Window::signal_frame_event( VMARG );

//FALCON_FUNC Window::signal_keys_changed( VMARG );

//FALCON_FUNC Window::signal_set_focus( VMARG );


/*#
    @method set_title gtk.Window
    @brief Set window title
    @param title Window title
 */
FALCON_FUNC Window::set_title( VMARG )
{
    Item* it = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !it || it->isNil() || !it->isString() )
    {
        throw_inv_params( "S" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString s( it->asString() );
    gtk_window_set_title( ((GtkWindow*)_obj), s.c_str() );
}


/*#
    @method set_resizable gtk.Window
    @brief Sets whether the user can resize a window.
    @param resizable true if the user can resize this window

    Windows are user resizable by default.
 */
FALCON_FUNC Window::set_resizable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_resizable( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_resizable gtk.Window
    @brief Gets the value set by set_resizable().
    @return (boolean)
 */
FALCON_FUNC Window::get_resizable( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_resizable( (GtkWindow*)_obj ) );
}


/*#
    @method activate_focus gtk.Window
    @brief Activates the current focused widget within the window.
    @return (boolean) true if a widget got activated.
 */
FALCON_FUNC Window::activate_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_activate_focus( (GtkWindow*)_obj ) );
}


/*#
    @method activate_default gtk.Window
    @brief Activates the default widget for the window
    @return (boolean) true if a widget got activated.

    Activates the default widget for the window, unless the current focused widget
    has been configured to receive the default action (see gtk_widget_set_receives_default()),
    in which case the focused widget is activated.
 */
FALCON_FUNC Window::activate_default( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_activate_default( (GtkWindow*)_obj ) );
}


/*#
    @method set_modal gtk.Window
    @brief Sets a window modal or non-modal.

    Modal windows prevent interaction with other windows in the same application.
    To keep modal dialogs on top of main application windows, use set_transient_for()
    to make the dialog transient for the parent; most window managers will then
    disallow lowering the dialog below the parent.
 */
FALCON_FUNC Window::set_modal( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_modal( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_default_size gtk.Window
    @brief Sets the default size of a window.
    @param width width in pixels, or -1 to unset the default width
    @param height height in pixels, or -1 to unset the default height

    If the window's "natural" size (its size request) is larger than the default,
    the default will be ignored. More generally, if the default size does not obey
    the geometry hints for the window (set_geometry_hints() can be used to set
    these explicitly), the default size will be clamped to the nearest permitted size.

    Unlike set_size_request(), which sets a size request for a widget and thus
    would keep users from shrinking the window, this function only sets the
    initial size, just as if the user had resized the window themselves.
    Users can still shrink the window again as they normally would. Setting a
    default size of -1 means to use the "natural" default size (the size request of the window).

    For more control over a window's initial size and how resizing works, investigate
    set_geometry_hints().

    For some uses, gtk_window_resize() is a more appropriate function. resize()
    changes the current size of the window, rather than the size to be used on
    initial display. gtk_window_resize() always affects the window itself, not
    the geometry widget.

    The default size of a window only affects the first time a window is shown;
    if a window is hidden and re-shown, it will remember the size it had prior
    to hiding, rather than using the default size.

    Windows can't actually be 0x0 in size, they must be at least 1x1, but passing
    0 for width and height is OK, resulting in a 1x1 default size.
 */
FALCON_FUNC Window::set_default_size( VMARG )
{
    Item* i_w = vm->param( 0 );
    Item* i_h = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || i_w->isNil() || !i_w->isInteger()
        || !i_h || i_h->isNil() || !i_h->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_default_size( (GtkWindow*)_obj, i_w->asInteger(), i_h->asInteger() );
}


//FALCON_FUNC set_geometry_hints( VMARG );


/*#
    @method set_gravity gtk.Window
    @brief Sets window gravity.
    @param gravity a valid GdkGravity value

    Window gravity defines the meaning of coordinates passed to move().
    See gtk_window_move() and GdkGravity for more details.

    The default window gravity is GDK_GRAVITY_NORTH_WEST which will typically
    "do what you mean."
 */
FALCON_FUNC Window::set_gravity( VMARG )
{
    Item* i_grav = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_grav || i_grav->isNil() || !i_grav->isInteger() )
        throw_inv_params( "GdkGravity" );
#endif
    int grav = i_grav->asInteger();
#ifndef NO_PARAMETER_CHECK
    if ( grav < 1 || grav > 10 )
        throw_inv_params( "GdkGravity" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_gravity( (GtkWindow*)_obj, (GdkGravity) grav );
}


/*#
    @method get_gravity gtk.Window
    @brief Gets the value set by set_gravity().
    @return (integer)
 */
FALCON_FUNC Window::get_gravity( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_window_get_gravity( (GtkWindow*)_obj ) );
}


/*#
    @method set_position gtk.Window
    @brief Sets a position constraint for this window.
    @param position a position constraint. (GtkWindowPosition)

    If the old or new constraint is GTK_WIN_POS_CENTER_ALWAYS, this will also
    cause the window to be repositioned to satisfy the new constraint.
 */
FALCON_FUNC Window::set_position( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || i_pos->isNil() || !i_pos->isInteger() )
        throw_inv_params( "GtkWindowPosition" );
#endif
    int pos = i_pos->asInteger();
#ifndef NO_PARAMETER_CHECK
    if ( pos < 0 || pos > 4 )
        throw_inv_params( "GtkWindowPosition" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_position( (GtkWindow*)_obj, (GtkWindowPosition) pos );
}


/*#
    @method set_transient_for gtk.Window
    @brief Sets the window transient for a parent window.
    @param parent (gtk.Window)

    Dialog windows should be set transient for the main application window they
    were spawned from. This allows window managers to e.g. keep the dialog on top
    of the main window, or center the dialog over the main window.
    gtk_dialog_new_with_buttons() and other convenience functions in GTK+ will
    sometimes call gtk_window_set_transient_for() on your behalf.

    Passing NULL for parent unsets the current transient window.

    On Windows, this function puts the child window on top of the parent, much as
    the window manager would have done on X.
 */
FALCON_FUNC Window::set_transient_for( VMARG )
{
    Item* i_win = vm->param( 0 );
    // this method accepts nil
#ifndef NO_PARAMETER_CHECK
    if ( i_win && !i_win->isNil() )
    {
        if ( !( i_win->isOfClass( "Window" ) || i_win->isOfClass( "gtk.Window" ) ) )
            throw_inv_params( "Window" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWindow* win = NULL;
    if ( i_win && !i_win->isNil() )
        win = (GtkWindow*)((GData*)i_win->asObject()->getUserData())->obj();
    gtk_window_set_transient_for( (GtkWindow*)_obj, win );
}


/*#
    @method set_destroy_with_parent gtk.Window
    @brief Destroys the window along with its parent.
    @param setting whether to destroy window with its transient parent (boolean)

    If setting is TRUE, then destroying the transient parent of window will also
    destroy window itself. This is useful for dialogs that shouldn't persist
    beyond the lifetime of the main window they're associated with, for example.
 */
FALCON_FUNC Window::set_destroy_with_parent( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_destroy_with_parent( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


//FALCON_FUNC Window::set_screen( VMARG );

//FALCON_FUNC Window::get_screen( VMARG );


/*#
    @method is_active gtk.Window
    @brief Returns whether the window is part of the current active toplevel.
    @return (boolean) true if the window part of the current active window.

    (That is, the toplevel window receiving keystrokes.) The return value is TRUE
    if the window is active toplevel itself, but also if it is, say, a GtkPlug
    embedded in the active toplevel. You might use this function if you wanted to
    draw a widget differently in an active window from a widget in an inactive window.
    See gtk_window_has_toplevel_focus()
 */
FALCON_FUNC Window::is_active( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_is_active( (GtkWindow*)_obj ) );
}


/*#
    @method has_topelevel_focus gtk.Window
    @brief Returns whether the input focus is within this GtkWindow.
    @return (boolean)

    For real toplevel windows, this is identical to gtk_window_is_active(), but
    for embedded windows, like GtkPlug, the results will differ.
 */
FALCON_FUNC Window::has_toplevel_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_has_toplevel_focus( (GtkWindow*)_obj ) );
}


//FALCON_FUNC Window::list_toplevels( VMARG );


/*#
    @method add_mnemonic gtk.Window
    @brief Adds a mnemonic to this window.
    @param keyval the mnemonic (integer)
    @param target the widget that gets activated by the mnemonic (gtk.Widget)
 */
FALCON_FUNC Window::add_mnemonic( VMARG )
{
    Item* i_keyval = vm->param( 0 );
    Item* i_target = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_keyval || i_keyval->isNil() || !i_keyval->isInteger()
        || !i_target || i_target->isNil() ||
        !( i_target->isOfClass( "Widget" ) || i_target->isOfClass( "gtk.Widget" ) ) )
        throw_inv_params( "I,Widget" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* target = (GtkWidget*)((GData*)i_target->asObject()->getUserData())->obj();
    gtk_window_add_mnemonic( (GtkWindow*)_obj, i_keyval->asInteger(), target );
}


/*#
    @method remove_mnemonic gtk.Window
    @brief Removes a mnemonic from this window.
    @param keyval the mnemonic (integer)
    @param target the widget that gets activated by the mnemonic (gtk.Widget)
 */
FALCON_FUNC Window::remove_mnemonic( VMARG )
{
    Item* i_keyval = vm->param( 0 );
    Item* i_target = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_keyval || i_keyval->isNil() || !i_keyval->isInteger()
        || !i_target || i_target->isNil() ||
        !( i_target->isOfClass( "Widget" ) || i_target->isOfClass( "gtk.Widget" ) ) )
        throw_inv_params( "I,Widget" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* target = (GtkWidget*)((GData*)i_target->asObject()->getUserData())->obj();
    gtk_window_remove_mnemonic( (GtkWindow*)_obj, i_keyval->asInteger(), target );
}


/*#
    @method mnemonic_activate gtk.Window
    @brief Activates the targets associated with the mnemonic.
    @param keyval the mnemonic
    @param modifier the modifiers (GdkModifierType)
    @return true if the activation is done.
 */
FALCON_FUNC Window::mnemonic_activate( VMARG )
{
    Item* i_keyval = vm->param( 0 );
    Item* i_modif = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_keyval || i_keyval->isNil() || !i_keyval->isInteger()
        || !i_modif || i_modif->isNil() || !i_modif->isInteger() )
        throw_inv_params( "I,GdkModifierType" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_mnemonic_activate( (GtkWindow*)_obj,
        i_keyval->asInteger(), (GdkModifierType) i_modif->asInteger() );
}


//FALCON_FUNC Window::activate_key( VMARG );

//FALCON_FUNC Window::propagate_key_event( VMARG );


/*#
    @method get_focus gtk.Window
    @brief Retrieves the current focused widget within the window.
    @return the currently focused widget, or nil if there is none.

    Note that this is the widget that would have the focus if the toplevel window
    focused; if the toplevel window is not focused then gtk_widget_has_focus (widget)
    will not be TRUE for the widget.
 */
FALCON_FUNC Window::get_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_window_get_focus( (GtkWindow*)_obj );
    if ( wdt )
    {
        Item* wki = vm->findWKI( "Widget" );
        vm->retval( new Gtk::Widget( wki->asClass(), wdt ) );
    }
    else
        vm->retnil();
}


/*#
    @method set_focus gtk.Window
    @brief Sets the focus on a widget.
    @param focus the focused widget (gtk.Widget)

    If focus is not the current focus widget, and is focusable, sets it as the
    focus widget for the window. If focus is NULL, unsets the focus widget for
    this window. To set the focus to a particular widget in the toplevel, it is
    usually more convenient to use grab_focus() instead of this function.
 */
FALCON_FUNC Window::set_focus( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    // this method accepts nil
#ifndef NO_PARAMETER_CHECK
    if ( i_wdt && !i_wdt->isNil() )
    {
        if ( !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
            throw_inv_params( "Widget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = NULL;
    if ( i_wdt && !i_wdt->isNil() )
        wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    gtk_window_set_focus( (GtkWindow*)_obj, wdt );
}


/*#
    @method get_default_widget gtk.Window
    @brief Returns the default widget for window.
    @return the default widget, or nil if there is none.
 */
FALCON_FUNC Window::get_default_widget( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_window_get_default_widget( (GtkWindow*)_obj );
    if ( wdt )
    {
        Item* wki = vm->findWKI( "Widget" );
        vm->retval( new Gtk::Widget( wki->asClass(), wdt ) );
    }
    else
        vm->retnil();
}


/*#
    @method set_default gtk.Window
    @brief Sets the default widget.
    @param default_widget (gtk.Widget)

    The default widget is the widget that's activated when the user presses Enter
    in a dialog (for example). This function sets or unsets the default widget for
    a GtkWindow about. When setting (rather than unsetting) the default widget it's
    generally easier to call gtk_widget_grab_focus() on the widget. Before making
    a widget the default widget, you must set the GTK_CAN_DEFAULT flag on the widget
    you'd like to make the default using GTK_WIDGET_SET_FLAGS().
 */
FALCON_FUNC Window::set_default( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() ||
        !( i_wdt->isOfClass( "Widget" ) || i_wdt->isOfClass( "gtk.Widget" ) ) )
            throw_inv_params( "Widget" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    gtk_window_set_default( (GtkWindow*)_obj, wdt );
}


/*#
    @method present gtk.Window
    @brief Presents a window to the user.

    This may mean raising the window in the stacking order, deiconifying it, moving
    it to the current desktop, and/or giving it the keyboard focus, possibly dependent
    on the user's platform, window manager, and preferences.

    If window is hidden, this function calls show() as well.

    This function should be used when the user tries to open a window that's already
    open. Say for example the preferences dialog is currently open, and the user
    chooses Preferences from the menu a second time; use present() to
    move the already-open dialog where the user can see it.

    If you are calling this function in response to a user interaction, it is
    preferable to use present_with_time().
 */
FALCON_FUNC Window::present( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_present( (GtkWindow*)_obj );
}


/*#
    @method present_with_time gtk.Window
    @brief Presents a window to the user in response to a user interaction.
    @param timestamp the timestamp of the user interaction (typically a button or key press event) which triggered this call

    If you need to present a window without a timestamp, use present().
 */
FALCON_FUNC Window::present_with_time( VMARG )
{
    Item* i_time = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_time || i_time->isNil() || !i_time->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_present_with_time( (GtkWindow*)_obj, i_time->asInteger() );
}


/*#
    @method iconify gtk.Window
    @brief Asks to iconify (i.e. minimize) the specified window.

    Note that you shouldn't assume the window is definitely iconified afterward,
    because other entities (e.g. the user or window manager) could deiconify it
    again, or there may not be a window manager in which case iconification isn't
    possible, etc. But normally the window will end up iconified. Just don't write
    code that crashes if not.

    It's permitted to call this function before showing a window, in which case
    the window will be iconified before it ever appears onscreen.

    You can track iconification via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::iconify( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_iconify( (GtkWindow*)_obj );
}


/*#
    @method deiconify gtk.Window
    @brief Asks to deiconify (i.e. unminimize) the specified window.

    Note that you shouldn't assume the window is definitely deiconified afterward,
    because other entities (e.g. the user or window manager) could iconify it again
    before your code which assumes deiconification gets to run.

    You can track iconification via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::deiconify( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_deiconify( (GtkWindow*)_obj );
}


/*#
    @method stick gtk.Window
    @brief Asks to stick window, which means that it will appear on all user desktops.

    Note that you shouldn't assume the window is definitely stuck afterward, because
    other entities (e.g. the user or window manager) could unstick it again, and
    some window managers do not support sticking windows. But normally the window
    will end up stuck. Just don't write code that crashes if not.

    It's permitted to call this function before showing a window.

    You can track stickiness via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::stick( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_stick( (GtkWindow*)_obj );
}


/*#
    @method unstick gtk.Window
    @brief Asks to unstick window, which means that it will appear on only one of the user's desktops.

    Note that you shouldn't assume the window is definitely unstuck afterward,
    because other entities (e.g. the user or window manager) could stick it again.
    But normally the window will end up stuck. Just don't write code that crashes if not.

    You can track stickiness via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::unstick( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_unstick( (GtkWindow*)_obj );
}


/*#
    @method maximize gtk.Window
    @brief Asks to maximize window, so that it becomes full-screen.

    Note that you shouldn't assume the window is definitely maximized afterward,
    because other entities (e.g. the user or window manager) could unmaximize it
    again, and not all window managers support maximization. But normally the
    window will end up maximized. Just don't write code that crashes if not.

    It's permitted to call this function before showing a window, in which case
    the window will be maximized when it appears onscreen initially.

    You can track maximization via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::maximize( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_maximize( (GtkWindow*)_obj );
}


/*#
    @method unmaximize gtk.Window
    @brief Asks to unmaximize window.

    Note that you shouldn't assume the window is definitely unmaximized afterward,
    because other entities (e.g. the user or window manager) could maximize it again,
    and not all window managers honor requests to unmaximize. But normally the window
    will end up unmaximized. Just don't write code that crashes if not.

    You can track maximization via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::unmaximize( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_unmaximize( (GtkWindow*)_obj );
}


/*#
    @method fullscreen gtk.Window
    @brief Asks to place window in the fullscreen state.

    Note that you shouldn't assume the window is definitely full screen afterward,
    because other entities (e.g. the user or window manager) could unfullscreen
    it again, and not all window managers honor requests to fullscreen windows.
    But normally the window will end up fullscreen. Just don't write code that
    crashes if not.

    You can track the fullscreen state via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::fullscreen( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_fullscreen( (GtkWindow*)_obj );
}


/*#
    @method unfullscreen gtk.Window
    @brief Asks to toggle off the fullscreen state for window.

    Note that you shouldn't assume the window is definitely not full screen afterward,
    because other entities (e.g. the user or window manager) could fullscreen it
    again, and not all window managers honor requests to unfullscreen windows.
    But normally the window will end up restored to its normal state.
    Just don't write code that crashes if not.

    You can track the fullscreen state via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::unfullscreen( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_unfullscreen( (GtkWindow*)_obj );
}


/*#
    @method set_keep_above gtk.Window
    @brief Asks to keep window above, so that it stays on top.
    @param setting whether to keep window above other windows (boolean)

    Note that you shouldn't assume the window is definitely above afterward, because
    other entities (e.g. the user or window manager) could not keep it above, and
    not all window managers support keeping windows above. But normally the window
    will end kept above. Just don't write code that crashes if not.

    It's permitted to call this function before showing a window, in which case
    the window will be kept above when it appears onscreen initially.

    You can track the above state via the "window-state-event" signal on GtkWidget.

    Note that, according to the Extended Window Manager Hints specification, the
    above state is mainly meant for user preferences and should not be used by
    applications e.g. for drawing attention to their dialogs.
 */
FALCON_FUNC Window::set_keep_above( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_keep_above( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_keep_below gtk.Window
    @brief Asks to keep window below, so that it stays in bottom.
    @param setting whether to keep window below other windows (boolean)

    Note that you shouldn't assume the window is definitely below afterward, because
    other entities (e.g. the user or window manager) could not keep it below, and
    not all window managers support putting windows below. But normally the window
    will be kept below. Just don't write code that crashes if not.

    It's permitted to call this function before showing a window, in which case
    the window will be kept below when it appears onscreen initially.

    You can track the below state via the "window-state-event" signal on GtkWidget.

    Note that, according to the Extended Window Manager Hints specification, the
    above state is mainly meant for user preferences and should not be used by
    applications e.g. for drawing attention to their dialogs.
 */
FALCON_FUNC Window::set_keep_below( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_keep_below( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method begin_resize_drag gtk.Window
    @brief Starts resizing a window.
    @param edge position of the resize control (integer, GdkWindowEdge)
    @param button mouse button that initiated the drag (integer)
    @param root_x X position where the user clicked to initiate the drag, in root window coordinates (integer)
    @param root_y Y position where the user clicked to initiate the drag (integer)
    @param timestamp timestamp from the click event that initiated the drag (integer)

    This function is used if an application has window resizing controls. When
    GDK can support it, the resize will be done using the standard mechanism for
    the window manager or windowing system. Otherwise, GDK will try to emulate
    window resizing, potentially not all that well, depending on the windowing system.
 */
FALCON_FUNC Window::begin_resize_drag( VMARG )
{
    Item* i_edge = vm->param( 0 );
    Item* i_button = vm->param( 1 );
    Item* i_root_x = vm->param( 2 );
    Item* i_root_y = vm->param( 3 );
    Item* i_tstamp = vm->param( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_edge || !i_button || !i_root_x || !i_root_y || !i_tstamp
        || !i_edge->isInteger() || !i_button->isInteger()
        || !i_root_x->isInteger() || !i_root_y->isInteger() || !i_tstamp->isInteger() )
        throw_inv_params( "I,I,I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_begin_resize_drag( (GtkWindow*)_obj, (GdkWindowEdge) i_edge->asInteger(),
        i_button->asInteger(), i_root_x->asInteger(), i_root_y->asInteger(),
        i_tstamp->asInteger() );
}


/*#
    @method begin_move_drag gtk.Window
    @brief Starts moving a window.
    @param button mouse button that initiated the drag (integer)
    @param root_x X position where the user clicked to initiate the drag, in root window coordinates (integer)
    @param root_y Y position where the user clicked to initiate the drag (integer)
    @param timestamp timestamp from the click event that initiated the drag (integer)

    This function is used if an application has window movement grips. When GDK
    can support it, the window movement will be done using the standard mechanism
    for the window manager or windowing system. Otherwise, GDK will try to emulate
    window movement, potentially not all that well, depending on the windowing system.
 */
FALCON_FUNC Window::begin_move_drag( VMARG )
{
    Item* i_button = vm->param( 0 );
    Item* i_root_x = vm->param( 1 );
    Item* i_root_y = vm->param( 2 );
    Item* i_tstamp = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_button || !i_root_x || !i_root_y || !i_tstamp
        || !i_button->isInteger() || !i_root_x->isInteger()
        || !i_root_y->isInteger() || !i_tstamp->isInteger() )
        throw_inv_params( "I,I,I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_begin_move_drag( (GtkWindow*)_obj, i_button->asInteger(),
        i_root_x->asInteger(), i_root_y->asInteger(), i_tstamp->asInteger() );
}


/*#
    @method set_decorated gtk.Window
    @brief Sets the window decorations.
    @param setting (boolean) true to decorate the window

    By default, windows are decorated with a title bar, resize controls, etc.
    Some window managers allow GTK+ to disable these decorations, creating a
    borderless window. If you set the decorated property to FALSE using this
    function, GTK+ will do its best to convince the window manager not to decorate
    the window. Depending on the system, this function may not have any effect
    when called on a window that is already visible, so you should call it before
    calling gtk_window_show().

    On Windows, this function always works, since there's no window manager policy involved.
 */
FALCON_FUNC Window::set_decorated( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_decorated( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_deletable gtk.Window
    @brief Sets the window deletable.
    @param setting (boolean) true to decorate the window as deletable.

    By default, windows have a close button in the window frame. Some window managers
    allow GTK+ to disable this button. If you set the deletable property to FALSE
    using this function, GTK+ will do its best to convince the window manager not
    to show a close button. Depending on the system, this function may not have any
    effect when called on a window that is already visible, so you should call it
    before calling gtk_window_show().

    On Windows, this function always works, since there's no window manager policy involved.
 */
FALCON_FUNC Window::set_deletable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_deletable( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_frame_dimensions gtk.Window
    @brief Sets the window frame dimensions.
    @param left The width of the left border (integer)
    @param top The height of the top border (integer)
    @param right The width of the right border (integer)
    @param bottom The height of the bottom border (integer)

    (Note: this is a special-purpose function intended for the framebuffer port;
    see gtk_window_set_has_frame(). It will have no effect on the window border
    drawn by the window manager, which is the normal case when using the X Window system.)

    For windows with frames (see gtk_window_set_has_frame()) this function can be
    used to change the size of the frame border.
 */
FALCON_FUNC Window::set_frame_dimensions( VMARG )
{
    Item* i_left = vm->param( 0 );
    Item* i_top = vm->param( 1 );
    Item* i_right = vm->param( 2 );
    Item* i_bottom = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_left || !i_top || !i_right || !i_bottom
        || !i_left->isInteger() || !i_top->isInteger()
        || !i_right->isInteger() || !i_bottom->isInteger() )
        throw_inv_params( "I,I,I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_frame_dimensions( (GtkWindow*)_obj, i_left->asInteger(),
        i_top->asInteger(), i_right->asInteger(), i_bottom->asInteger() );
}


/*#
    @method set_has_frame gtk.Window
    @brief Sets the window frame.
    @param setting (boolean)

    (Note: this is a special-purpose function for the framebuffer port, that causes
    GTK+ to draw its own window border. For most applications, you want set_decorated()
    instead, which tells the window manager whether to draw the window border.)

    If this function is called on a window with setting of TRUE, before it is
    realized or showed, it will have a "frame" window around window->window,
    accessible in window->frame. Using the signal frame_event you can receive all
    events targeted at the frame.

    This function is used by the linux-fb port to implement managed windows, but
    it could conceivably be used by X-programs that want to do their own window decorations.
 */
FALCON_FUNC Window::set_has_frame( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_has_frame( (GtkWindow*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


//FALCON_FUNC Window::set_mnemonic_modifier( VMARG );

//FALCON_FUNC Window::set_type_hint( VMARG );

//FALCON_FUNC Window::set_skip_taskbar_hint( VMARG );

//FALCON_FUNC Window::set_skip_pager_hint( VMARG );

//FALCON_FUNC Window::set_urgency_hint( VMARG );

//FALCON_FUNC Window::set_accept_focus( VMARG );

//FALCON_FUNC Window::set_focus_on_map( VMARG );

//FALCON_FUNC Window::set_startup_id( VMARG );

//FALCON_FUNC Window::set_role( VMARG );

//FALCON_FUNC Window::get_decorated( VMARG );

//FALCON_FUNC Window::get_deletable( VMARG );

//FALCON_FUNC Window::get_default_icon_list( VMARG );

//FALCON_FUNC Window::get_default_icon_name( VMARG );

//FALCON_FUNC Window::get_default_size( VMARG );

//FALCON_FUNC Window::get_destroy_with_parent( VMARG );

//FALCON_FUNC Window::get_frame_dimensions( VMARG );

//FALCON_FUNC Window::get_has_frame( VMARG );

//FALCON_FUNC Window::get_icon( VMARG );

//FALCON_FUNC Window::get_icon_list( VMARG );

//FALCON_FUNC Window::get_icon_name( VMARG );

//FALCON_FUNC Window::get_mnemonic_modifier( VMARG );

//FALCON_FUNC Window::get_modal( VMARG );

//FALCON_FUNC Window::get_position( VMARG );

//FALCON_FUNC Window::get_role( VMARG );

//FALCON_FUNC Window::get_size( VMARG );


/*#
    @method get_title gtk.Window
    @brief Get window title
 */
FALCON_FUNC Window::get_title( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* t = gtk_window_get_title( ((GtkWindow*)_obj) );
    vm->retval( t ? new String( t ) : new String() );
}


//FALCON_FUNC Window::get_transient_for( VMARG );

//FALCON_FUNC Window::get_type_hint( VMARG );

//FALCON_FUNC Window::get_skip_taskbar_hint( VMARG );

//FALCON_FUNC Window::get_skip_pager_hint( VMARG );

//FALCON_FUNC Window::get_urgency_hint( VMARG );

//FALCON_FUNC Window::get_accept_focus( VMARG );

//FALCON_FUNC Window::get_focus_on_map( VMARG );

//FALCON_FUNC Window::get_group( VMARG );

//FALCON_FUNC Window::get_window_type( VMARG );

//FALCON_FUNC Window::move( VMARG );

//FALCON_FUNC Window::parse_geometry( VMARG );

//FALCON_FUNC Window::reshow_with_initial_size( VMARG );

//FALCON_FUNC Window::resize( VMARG );

//FALCON_FUNC Window::set_default_icon_list( VMARG );

//FALCON_FUNC Window::set_default_icon( VMARG );

//FALCON_FUNC Window::set_default_icon_from_file( VMARG );

//FALCON_FUNC Window::set_default_icon_name( VMARG );

//FALCON_FUNC Window::set_icon( VMARG );

//FALCON_FUNC Window::set_icon_list( VMARG );

//FALCON_FUNC Window::set_icon_from_file( VMARG );

//FALCON_FUNC Window::set_icon_name( VMARG );

//FALCON_FUNC Window::set_auto_startup_notification( VMARG );

//FALCON_FUNC Window::get_opacity( VMARG );

//FALCON_FUNC Window::set_opacity( VMARG );

//FALCON_FUNC Window::get_mnemonics_visible( VMARG );

//FALCON_FUNC Window::set_mnemonics_visible( VMARG );


} // Gtk
} // Falcon
