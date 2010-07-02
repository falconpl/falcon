/**
 *  \file gtk_Window.cpp
 */

#include "gtk_Window.hpp"

//#include "gdk_Event.hpp"
#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Window::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Window = mod->addClass( "GtkWindow", &Window::init )
        ->addParam( "type" );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_Window->getClassDef()->addInheritance( in );

    c_Window->setWKS( true );
    c_Window->getClassDef()->factory( &Window::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_activate_default",&Window::signal_activate_default },
    { "signal_activate_focus",&Window::signal_activate_focus },
    //{ "signal_frame_event",&Window::signal_frame_event },
    { "signal_keys_changed",&Window::signal_keys_changed },
    { "signal_set_focus",   &Window::signal_set_focus },
    { "set_title",          &Window::set_title },
    { "set_wmclass",        &Window::set_wmclass },
#if 0 // deprecated
    { "set_policy",         &Window::set_policy },
#endif
    { "set_resizable",      &Window::set_resizable },
    { "get_resizable",      &Window::get_resizable },
    //{ "add_accel_group",    &Window::add_accel_group },
    //{ "remove_accel_group", &Window::remove_accel_group },
#if 0 // deprecated
    { "position",           &Window::position },
#endif
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
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_default_widget", &Window::get_default_widget },
#endif
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
    { "set_mnemonic_modifier",&Window::set_mnemonic_modifier },
    { "set_type_hint",      &Window::set_type_hint },
    { "set_skip_taskbar_hint",&Window::set_skip_taskbar_hint },
    { "set_skip_pager_hint",&Window::set_skip_pager_hint },
    { "set_urgency_hint",   &Window::set_urgency_hint },
    { "set_accept_focus",   &Window::set_accept_focus },
    { "set_focus_on_map",   &Window::set_focus_on_map },
    { "set_startup_id",     &Window::set_startup_id },
    { "set_role",           &Window::set_role },
    { "get_decorated",      &Window::get_decorated },
    { "get_deletable",      &Window::get_deletable },
    //{ "get_default_icon_list",        &Window::set_gravity },
#if GTK_CHECK_VERSION( 2, 16, 0 )
    { "get_default_icon_name",&Window::get_default_icon_name },
#endif
    { "get_default_size",   &Window::get_default_size },
    { "get_destroy_with_parent",&Window::get_destroy_with_parent },
    { "get_frame_dimensions",&Window::get_frame_dimensions },
    { "get_has_frame",      &Window::get_has_frame },
    //{ "get_icon",        &Window::set_gravity },
    //{ "get_icon_list",        &Window::set_gravity },
    { "get_icon_name",      &Window::get_icon_name },
    { "get_mnemonic_modifier",&Window::get_mnemonic_modifier },
    { "get_modal",          &Window::get_modal },
    { "get_position",       &Window::get_position },
    { "get_role",           &Window::get_role },
    { "get_size",           &Window::get_size },
    { "get_title",          &Window::get_title },
    { "get_transient_for",  &Window::get_transient_for },
    { "get_type_hint",      &Window::get_type_hint },
    { "get_skip_taskbar_hint",&Window::get_skip_taskbar_hint },
    { "get_skip_pager_hint",&Window::get_skip_pager_hint },
    { "get_urgency_hint",   &Window::get_urgency_hint },
    { "get_accept_focus",   &Window::get_accept_focus },
    { "get_focus_on_map",   &Window::get_focus_on_map },
    //{ "get_group",        &Window::set_gravity },
#if GTK_CHECK_VERSION( 2, 20, 0 )
    { "get_window_type",    &Window::get_window_type },
#endif
    { "move",               &Window::move },
    { "parse_geometry",     &Window::parse_geometry },
    { "reshow_with_initial_size",&Window::reshow_with_initial_size },
    { "resize",             &Window::resize },
    //{ "set_default_icon_list",        &Window::set_gravity },
    //{ "set_default_icon",        &Window::set_gravity },
    //{ "set_default_icon_from_file",        &Window::set_gravity },
    { "set_default_icon_name",&Window::set_default_icon_name },
    //{ "set_icon",        &Window::set_gravity },
    //{ "set_icon_list",        &Window::set_gravity },
    //{ "set_icon_from_file",        &Window::set_gravity },
    { "set_icon_name",        &Window::set_icon_name },
    { "set_auto_startup_notification",&Window::set_auto_startup_notification },
    { "get_opacity",        &Window::get_opacity },
    { "set_opacity",        &Window::set_opacity },
    //{ "get_mnemonics_visible",        &Window::set_gravity },
    //{ "set_mnemonics_visible",        &Window::set_gravity },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Window, meth->name, meth->cb );
}


Window::Window( const Falcon::CoreClass* gen, const GtkWindow* win )
    :
    Gtk::CoreGObject( gen, (GObject*) win )
{}


Falcon::CoreObject* Window::factory( const Falcon::CoreClass* gen, void* win, bool )
{
    return new Window( gen, (GtkWindow*) win );
}


/*#
    @class GtkWindow
    @optparam type GTK_WINDOW_TOPLEVEL (default) or GTK_WINDOW_POPUP

    Toplevel which can contain other widgets.

    Creates a new GtkWindow, which is a toplevel window that can contain other
    widgets. Nearly always, the type of the window should be GTK_WINDOW_TOPLEVEL.
    If you're implementing something like a popup menu from scratch (which is
    a bad idea, just use GtkMenu), you might use GTK_WINDOW_POPUP.
    GTK_WINDOW_POPUP is not for dialogs, though in some other toolkits dialogs
    are called "popups". In GTK+, GTK_WINDOW_POPUP means a pop-up menu or pop-up
    tooltip. On X11, popup windows are not controlled by the window manager.
 */
FALCON_FUNC Window::init( VMARG )
{
    MYSELF;
    if ( self->getObject() )
        return;

    Item* i_wtype = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_wtype && !i_wtype->isInteger() )
        throw_inv_params( "GtkWindowType" );
#endif
    GtkWindowType gwt = i_wtype ? (GtkWindowType) i_wtype->asInteger()
                        : GTK_WINDOW_TOPLEVEL;
    self->setObject( (GObject*) gtk_window_new( gwt ) );
}


/*#
    @method signal_activate_default GtkWindow
    @brief The ::activate-default signal is a keybinding signal which gets emitted when the user activates the default widget of window.
 */
FALCON_FUNC Window::signal_activate_default( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "activate_default", (void*) &Window::on_activate_default, vm );
}


void Window::on_activate_default( GtkWindow* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "activate_default",
                               "on_activate_default", (VMachine*)_vm );
}


/*#
    @method signal_activate_focus GtkWindow
    @brief The activate-focus signal is a keybinding signal which gets emitted when the user activates the currently focused widget of window.
 */
FALCON_FUNC Window::signal_activate_focus( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "activate_focus", (void*) &Window::on_activate_focus, vm );
}


void Window::on_activate_focus( GtkWindow* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "activate_focus",
                               "on_activate_focus", (VMachine*)_vm );
}


//FALCON_FUNC Window::signal_frame_event( VMARG );

//void Window::on_frame_event( GtkWindow* obj, GdkEvent* ev, gpointer _vm );


/*#
    @method signal_keys_changed GtkWindow
    @brief The keys-changed signal gets emitted when the set of accelerators or mnemonics that are associated with window changes.
 */
FALCON_FUNC Window::signal_keys_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "keys_changed", (void*) &Window::on_keys_changed, vm );
}


void Window::on_keys_changed( GtkWindow* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "keys_changed",
                               "on_keys_changed", (VMachine*)_vm );
}


/*#
    @method signal_set_focus GtkWindow
    @brief .
 */
FALCON_FUNC Window::signal_set_focus( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "set_focus", (void*) &Window::on_set_focus, vm );
}


void Window::on_set_focus( GtkWindow* obj, GtkWidget* wdt, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "set_focus", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkWidget" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_set_focus", it ) )
            {
                printf(
                "[GtkWindow::on_set_focus] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::Widget( wki->asClass(), wdt ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method set_title GtkWindow
    @brief Sets the title of the GtkWindow.
    @param title title of the window

    The title of a window will be displayed in its title bar; on the X Window
    System, the title bar is rendered by the window manager, so exactly how the
    title appears to users may vary according to a user's exact configuration.
    The title should help a user distinguish this window from other windows they
    may have open. A good title might include the application name and current
    document filename, for example.
 */
FALCON_FUNC Window::set_title( VMARG )
{
    Item* i_title = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_title || !i_title->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString title( i_title->asString() );
    gtk_window_set_title( (GtkWindow*)_obj, title.c_str() );
}


/*#
    @method set_wmclass GtkWindow
    @brief Don't use this function.
    @param wmclass_name window name hint
    @param wmclass_class window class hint

    It sets the X Window System "class" and "name" hints for a window. According
    to the ICCCM, you should always set these to the same value for all windows
    in an application, and GTK+ sets them to that value by default, so calling
    this function is sort of pointless. However, you may want to call
    gtk_window_set_role() on each window in your application, for the benefit of
    the session manager. Setting the role allows the window manager to restore
    window positions when loading a saved session.
 */
FALCON_FUNC Window::set_wmclass( VMARG )
{
    Item* i_nm = vm->param( 0 );
    Item* i_cl = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_nm || !i_nm->isString()
        || !i_cl || !i_cl->isString() )
        throw_inv_params( "S,S" );
#endif
    AutoCString nm( i_nm->asString() );
    AutoCString cl( i_cl->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_wmclass( (GtkWindow*)_obj, nm.c_str(), cl.c_str() );
}


#if 0 // deprecated
FALCON_FUNC Window::set_policy( VMARG );
#endif


/*#
    @method set_resizable GtkWindow
    @brief Sets whether the user can resize a window.
    @param resizable true if the user can resize this window

    Windows are user resizable by default.
 */
FALCON_FUNC Window::set_resizable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_resizable( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_resizable GtkWindow
    @brief Gets the value set by set_resizable().
    @return TRUE if the user can resize the window
 */
FALCON_FUNC Window::get_resizable( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_resizable( (GtkWindow*)_obj ) );
}


//FALCON_FUNC Window::add_accel_group( VMARG );

//FALCON_FUNC Window::remove_accel_group( VMARG );


#if 0 // deprecated
FALCON_FUNC Window::position( VMARG );
#endif


/*#
    @method activate_focus GtkWindow
    @brief Activates the current focused widget within the window.
    @return (boolean) true if a widget got activated.
 */
FALCON_FUNC Window::activate_focus( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_activate_focus( (GtkWindow*)_obj ) );
}


/*#
    @method activate_default GtkWindow
    @brief Activates the default widget for the window
    @return (boolean) true if a widget got activated.

    Activates the default widget for the window, unless the current focused widget
    has been configured to receive the default action (see gtk_widget_set_receives_default()),
    in which case the focused widget is activated.
 */
FALCON_FUNC Window::activate_default( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_activate_default( (GtkWindow*)_obj ) );
}


/*#
    @method set_modal GtkWindow
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
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_modal( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_default_size GtkWindow
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

    For some uses, resize() is a more appropriate function. resize()
    changes the current size of the window, rather than the size to be used on
    initial display. resize() always affects the window itself, not
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
    if ( !i_w || !i_w->isInteger()
        || !i_h || !i_h->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_default_size( (GtkWindow*)_obj, i_w->asInteger(), i_h->asInteger() );
}


//FALCON_FUNC set_geometry_hints( VMARG );


/*#
    @method set_gravity GtkWindow
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
    if ( !i_grav || !i_grav->isInteger() )
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
    @method get_gravity GtkWindow
    @brief Gets the value set by set_gravity().
    @return (integer)
 */
FALCON_FUNC Window::get_gravity( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_window_get_gravity( (GtkWindow*)_obj ) );
}


/*#
    @method set_position GtkWindow
    @brief Sets a position constraint for this window.
    @param position a position constraint. (GtkWindowPosition)

    If the old or new constraint is GTK_WIN_POS_CENTER_ALWAYS, this will also
    cause the window to be repositioned to satisfy the new constraint.
 */
FALCON_FUNC Window::set_position( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
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
    @method set_transient_for GtkWindow
    @brief Sets the window transient for a parent window.
    @param parent parent window, or NULL.

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
#ifndef NO_PARAMETER_CHECK
    if ( !i_win || !( i_win->isNil() || ( i_win->isObject()
        && IS_DERIVED( i_win, GtkWindow ) ) ) )
        throw_inv_params( "[GtkWindow]" );
#endif
    GtkWindow* win = i_win->isNil() ? NULL
                    : (GtkWindow*) COREGOBJECT( i_win )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_transient_for( (GtkWindow*)_obj, win );
}


/*#
    @method set_destroy_with_parent GtkWindow
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
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_destroy_with_parent( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


//FALCON_FUNC Window::set_screen( VMARG );

//FALCON_FUNC Window::get_screen( VMARG );


/*#
    @method is_active GtkWindow
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
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_is_active( (GtkWindow*)_obj ) );
}


/*#
    @method has_topelevel_focus GtkWindow
    @brief Returns whether the input focus is within this GtkWindow.
    @return (boolean)

    For real toplevel windows, this is identical to gtk_window_is_active(), but
    for embedded windows, like GtkPlug, the results will differ.
 */
FALCON_FUNC Window::has_toplevel_focus( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_has_toplevel_focus( (GtkWindow*)_obj ) );
}


//FALCON_FUNC Window::list_toplevels( VMARG );


/*#
    @method add_mnemonic GtkWindow
    @brief Adds a mnemonic to this window.
    @param keyval the mnemonic character
    @param target the widget that gets activated by the mnemonic (GtkWidget)
 */
FALCON_FUNC Window::add_mnemonic( VMARG )
{
    Item* i_keyval = vm->param( 0 );
    Item* i_target = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_keyval || !i_keyval->isString()
        || !i_target || !i_target->isObject() || !IS_DERIVED( i_target, GtkWidget ) )
        throw_inv_params( "S,GtkWidget" );
#endif
    String* chr = i_keyval->asString();
    guint keyval = chr->length() ? chr->getCharAt( 0 ) : 0;
    GtkWidget* target = (GtkWidget*) COREGOBJECT( i_target )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_window_add_mnemonic( (GtkWindow*)_obj, keyval, target );
}


/*#
    @method remove_mnemonic GtkWindow
    @brief Removes a mnemonic from this window.
    @param keyval the mnemonic character
    @param target the widget that gets activated by the mnemonic (GtkWidget)
 */
FALCON_FUNC Window::remove_mnemonic( VMARG )
{
    Item* i_keyval = vm->param( 0 );
    Item* i_target = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_keyval || !i_keyval->isString()
        || !i_target || !i_target->isObject() || !IS_DERIVED( i_target, GtkWidget ) )
        throw_inv_params( "S,GtkWidget" );
#endif
    String* chr = i_keyval->asString();
    guint keyval = chr->length() ? chr->getCharAt( 0 ) : 0;
    GtkWidget* target = (GtkWidget*) COREGOBJECT( i_target )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_window_remove_mnemonic( (GtkWindow*)_obj, keyval, target );
}


/*#
    @method mnemonic_activate GtkWindow
    @brief Activates the targets associated with the mnemonic.
    @param keyval the mnemonic character
    @param modifier the modifiers (GdkModifierType)
    @return true if the activation is done.
 */
FALCON_FUNC Window::mnemonic_activate( VMARG )
{
    Item* i_keyval = vm->param( 0 );
    Item* i_modif = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_keyval || !i_keyval->isString()
        || !i_modif || !i_modif->isInteger() )
        throw_inv_params( "S,GdkModifierType" );
#endif
    String* chr = i_keyval->asString();
    guint keyval = chr->length() ? chr->getCharAt( 0 ) : 0;
    MYSELF;
    GET_OBJ( self );
    gtk_window_mnemonic_activate( (GtkWindow*)_obj,
                                  keyval, (GdkModifierType) i_modif->asInteger() );
}


//FALCON_FUNC Window::activate_key( VMARG );

//FALCON_FUNC Window::propagate_key_event( VMARG );


/*#
    @method get_focus GtkWindow
    @brief Retrieves the current focused widget within the window.
    @return the currently focused widget, or nil if there is none.

    Note that this is the widget that would have the focus if the toplevel window
    focused; if the toplevel window is not focused then gtk_widget_has_focus (widget)
    will not be TRUE for the widget.
 */
FALCON_FUNC Window::get_focus( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_window_get_focus( (GtkWindow*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method set_focus GtkWindow
    @brief Sets the focus on a widget.
    @param focus widget to be the new focus widget, or NULL to unset any focus widget for the toplevel window.

    If focus is not the current focus widget, and is focusable, sets it as the
    focus widget for the window. If focus is NULL, unsets the focus widget for
    this window. To set the focus to a particular widget in the toplevel, it is
    usually more convenient to use grab_focus() instead of this function.
 */
FALCON_FUNC Window::set_focus( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !( i_wdt->isNil() || ( i_wdt->isObject()
        && IS_DERIVED( i_wdt, GtkWidget ) ) ) )
        throw_inv_params( "[GtkWidget]" );
#endif
    GtkWidget* wdt = i_wdt->isNil() ? NULL
                    : (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_focus( (GtkWindow*)_obj, wdt );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_default_widget GtkWindow
    @brief Returns the default widget for window.
    @return the default widget, or nil if there is none.
 */
FALCON_FUNC Window::get_default_widget( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_window_get_default_widget( (GtkWindow*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}
#endif


/*#
    @method set_default GtkWindow
    @brief Sets the default widget.
    @param default_widget widget to be the default, or NULL to unset the default widget for the toplevel.

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
    if ( !i_wdt || !( i_wdt->isNil() || ( i_wdt->isObject()
        && IS_DERIVED( i_wdt, GtkWidget ) ) ) )
        throw_inv_params( "[GtkWidget]" );
#endif
    GtkWidget* wdt = i_wdt->isNil() ? NULL
                    : (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_default( (GtkWindow*)_obj, wdt );
}


/*#
    @method present GtkWindow
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
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_present( (GtkWindow*)_obj );
}


/*#
    @method present_with_time GtkWindow
    @brief Presents a window to the user in response to a user interaction.
    @param timestamp the timestamp of the user interaction (typically a button or key press event) which triggered this call

    If you need to present a window without a timestamp, use present().
 */
FALCON_FUNC Window::present_with_time( VMARG )
{
    Item* i_time = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_time || !i_time->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_present_with_time( (GtkWindow*)_obj, i_time->asInteger() );
}


/*#
    @method iconify GtkWindow
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
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_iconify( (GtkWindow*)_obj );
}


/*#
    @method deiconify GtkWindow
    @brief Asks to deiconify (i.e. unminimize) the specified window.

    Note that you shouldn't assume the window is definitely deiconified afterward,
    because other entities (e.g. the user or window manager) could iconify it again
    before your code which assumes deiconification gets to run.

    You can track iconification via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::deiconify( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_deiconify( (GtkWindow*)_obj );
}


/*#
    @method stick GtkWindow
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
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_stick( (GtkWindow*)_obj );
}


/*#
    @method unstick GtkWindow
    @brief Asks to unstick window, which means that it will appear on only one of the user's desktops.

    Note that you shouldn't assume the window is definitely unstuck afterward,
    because other entities (e.g. the user or window manager) could stick it again.
    But normally the window will end up stuck. Just don't write code that crashes if not.

    You can track stickiness via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::unstick( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_unstick( (GtkWindow*)_obj );
}


/*#
    @method maximize GtkWindow
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
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_maximize( (GtkWindow*)_obj );
}


/*#
    @method unmaximize GtkWindow
    @brief Asks to unmaximize window.

    Note that you shouldn't assume the window is definitely unmaximized afterward,
    because other entities (e.g. the user or window manager) could maximize it again,
    and not all window managers honor requests to unmaximize. But normally the window
    will end up unmaximized. Just don't write code that crashes if not.

    You can track maximization via the "window-state-event" signal on GtkWidget.
 */
FALCON_FUNC Window::unmaximize( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_unmaximize( (GtkWindow*)_obj );
}


/*#
    @method fullscreen GtkWindow
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
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_fullscreen( (GtkWindow*)_obj );
}


/*#
    @method unfullscreen GtkWindow
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
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_unfullscreen( (GtkWindow*)_obj );
}


/*#
    @method set_keep_above GtkWindow
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
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_keep_above( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_keep_below GtkWindow
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
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_keep_below( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method begin_resize_drag GtkWindow
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
        throw_inv_params( "I,I,I,I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_begin_resize_drag( (GtkWindow*)_obj,
                                  (GdkWindowEdge) i_edge->asInteger(),
                                  i_button->asInteger(),
                                  i_root_x->asInteger(),
                                  i_root_y->asInteger(),
                                  i_tstamp->asInteger() );
}


/*#
    @method begin_move_drag GtkWindow
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
    gtk_window_begin_move_drag( (GtkWindow*)_obj,
                                i_button->asInteger(),
                                i_root_x->asInteger(),
                                i_root_y->asInteger(),
                                i_tstamp->asInteger() );
}


/*#
    @method set_decorated GtkWindow
    @brief Sets the window decorations.
    @param setting (boolean) true to decorate the window

    By default, windows are decorated with a title bar, resize controls, etc.
    Some window managers allow GTK+ to disable these decorations, creating a
    borderless window. If you set the decorated property to FALSE using this
    function, GTK+ will do its best to convince the window manager not to decorate
    the window. Depending on the system, this function may not have any effect
    when called on a window that is already visible, so you should call it before
    calling gtk_window_show().

    On Windows, this function always works, since there's no window manager
    policy involved.
 */
FALCON_FUNC Window::set_decorated( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_decorated( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_deletable GtkWindow
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
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_deletable( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_frame_dimensions GtkWindow
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
    gtk_window_set_frame_dimensions( (GtkWindow*)_obj,
                                     i_left->asInteger(),
                                     i_top->asInteger(),
                                     i_right->asInteger(),
                                     i_bottom->asInteger() );
}


/*#
    @method set_has_frame GtkWindow
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
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_has_frame( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_mnemonic_modifier GtkWindow
    @brief Sets the mnemonic modifier for this window.
    @param modifier the modifier mask used to activate mnemonics on this window. (GdkModifierType)
 */
FALCON_FUNC Window::set_mnemonic_modifier( VMARG )
{
    Item* i_modif = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_modif || !i_modif->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_mnemonic_modifier( (GtkWindow*)_obj,
                                      (GdkModifierType) i_modif->asInteger() );
}


/*#
    @method set_type_hint GtkWindow
    @brief Sets the window type hint.
    @param hint the window type (GdkWindowTypeHint)

    By setting the type hint for the window, you allow the window manager to decorate
    and handle the window in a way which is suitable to the function of the window
    in your application.

    This function should be called before the window becomes visible.
 */
FALCON_FUNC Window::set_type_hint( VMARG )
{
    Item* i_hint = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_hint || !i_hint->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_type_hint( (GtkWindow*)_obj,
                              (GdkWindowTypeHint) i_hint->asInteger() );
}


/*#
    @method set_skip_taskbar_hint GtkWindow
    @brief Sets the window taskbar hint.
    @param setting (boolean) true to keep this window from appearing in the task bar

    Windows may set a hint asking the desktop environment not to display the window
    in the task bar. This function sets this hint.
 */
FALCON_FUNC Window::set_skip_taskbar_hint( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_skip_taskbar_hint( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_skip_pager_hint GtkWindow
    @brief Sets the window pager hint.
    @param setting (boolean) true to keep this window from appearing in the pager

    Windows may set a hint asking the desktop environment not to display the window
    in the pager. This function sets this hint. (A "pager" is any desktop navigation
    tool such as a workspace switcher that displays a thumbnail representation of
    the windows on the screen.)
 */
FALCON_FUNC Window::set_skip_pager_hint( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_skip_pager_hint( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_urgency_hint GtkWindow
    @brief Sets the window urgency hint.
    @param setting (boolean) true to mark this window as urgent

    Windows may set a hint asking the desktop environment to draw the users attention
    to the window. This function sets this hint.
 */
FALCON_FUNC Window::set_urgency_hint( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_urgency_hint( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_accept_focus GtkWindow
    @brief Sets the window accept-focus hint.
    @param setting (boolean) True to let this window receive input focus

    Windows may set a hint asking the desktop environment not to receive the input
    focus. This function sets this hint.
 */
FALCON_FUNC Window::set_accept_focus( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_accept_focus( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_focus_on_map GtkWindow
    @brief Sets the window focus-on-map hint.
    @param setting (boolean) true to let this window receive input focus on map

    Windows may set a hint asking the desktop environment not to receive the input
    focus when the window is mapped. This function sets this hint.
 */
FALCON_FUNC Window::set_focus_on_map( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_focus_on_map( (GtkWindow*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method set_startup_id GtkWindow
    @brief Sets the window startup identifier.
    @param startup_id a string with startup-notification identifier

    Startup notification identifiers are used by desktop environment to track
    application startup, to provide user feedback and other features. This function
    changes the corresponding property on the underlying GdkWindow. Normally,
    startup identifier is managed automatically and you should only use this
    function in special cases like transferring focus from other processes.
    You should use this function before calling present() or any
    equivalent function generating a window map event.

    This function is only useful on X11, not with other GTK+ targets.
 */
FALCON_FUNC Window::set_startup_id( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || !i_id->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString id( i_id->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_startup_id( (GtkWindow*)_obj, id.c_str() );
}


/*#
    @method set_role GtkWindow
    @brief Sets the window role.
    @param role unique identifier for the window to be used when restoring a session (string)

    This function is only useful on X11, not with other GTK+ targets.

    In combination with the window title, the window role allows a window manager
    to identify "the same" window when an application is restarted. So for example
    you might set the "toolbox" role on your app's toolbox window, so that when the
    user restarts their session, the window manager can put the toolbox back in
    the same place.

    If a window already has a unique title, you don't need to set the role, since
    the WM can use the title to identify the window when restoring the session.
 */
FALCON_FUNC Window::set_role( VMARG )
{
    Item* i_role = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_role || !i_role->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString role( i_role->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_role( (GtkWindow*)_obj, role.c_str() );
}


/*#
    @method get_decorated GtkWindow
    @brief Returns whether the window has been set to have decorations such as a title bar via set_decorated().
    @return (boolean) true if the window has been set to have decorations
 */
FALCON_FUNC Window::get_decorated( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_decorated( (GtkWindow*)_obj ) );
}


/*#
    @method get_deletable GtkWindow
    @brief Returns whether the window has been set to have a close button via set_deletable().
    @return (boolean) true if the window has been set to have a close button
 */
FALCON_FUNC Window::get_deletable( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_deletable( (GtkWindow*)_obj ) );
}


//FALCON_FUNC Window::get_default_icon_list( VMARG );


#if GTK_CHECK_VERSION( 2, 16, 0 )
/*#
    @method get_default_icon_name GtkWindow
    @brief Gets the window default icon name.
    @return (string) the fallback icon name for windows

    Returns the fallback icon name for windows that has been set with set_default_icon_name().
    The returned string is owned by GTK+ and should not be modified. It is only valid
    until the next call to set_default_icon_name().
 */
FALCON_FUNC Window::get_default_icon_name( VMARG )
{
    NO_ARGS
    const gchar* nam = gtk_window_get_default_icon_name();
    vm->retval( nam ? UTF8String( nam ) : UTF8String( "" ) );
}
#endif


/*#
    @method get_default_size GtkWindow
    @brief Gets the default size of the window.
    @return an array [ default width, default height ]

    A value of -1 for the width or height indicates that a default size has not
    been explicitly set for that dimension, so the "natural" size of the window
    will be used.
 */
FALCON_FUNC Window::get_default_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint w, h;
    gtk_window_get_default_size( (GtkWindow*)_obj, &w, &h );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method get_destroy_with_parent GtkWindow
    @brief Returns whether the window will be destroyed with its transient parent.
    @return (boolean) true if the window will be destroyed with its transient parent.
 */
FALCON_FUNC Window::get_destroy_with_parent( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_destroy_with_parent( (GtkWindow*)_obj ) );
}


/*#
    @method get_frame_dimensions GtkWindow
    @brief Retrieves the dimensions of the frame window for this toplevel.
    @return an array [ left, top, right, bottom ]

    (Note: this is a special-purpose function intended for the framebuffer port;
    see set_has_frame(). It will not return the size of the window border drawn
    by the window manager, which is the normal case when using a windowing system.
    See gdk_window_get_frame_extents() to get the standard window border extents.)

    See gtk_window_set_has_frame(), gtk_window_set_frame_dimensions().
 */
FALCON_FUNC Window::get_frame_dimensions( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint l, t, r, b;
    gtk_window_get_frame_dimensions( (GtkWindow*)_obj, &l, &t, &r, &b );
    CoreArray* arr = new CoreArray( 4 );
    arr->append( l );
    arr->append( t );
    arr->append( r );
    arr->append( b );
    vm->retval( arr );
}


/*#
    @method get_has_frame GtkWindow
    @brief Accessor for whether the window has a frame window exterior to window->window.
    @return (boolean) true if a frame has been added to the window via set_has_frame().

    Gets the value set by set_has_frame().
 */
FALCON_FUNC Window::get_has_frame( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_has_frame( (GtkWindow*)_obj ) );
}


//FALCON_FUNC Window::get_icon( VMARG );

//FALCON_FUNC Window::get_icon_list( VMARG );


/*#
    @method get_icon_name GtkWindow
    @brief Returns the name of the themed icon for the window, see set_icon_name().
    @return the icon name or nil if the window has no themed icon
 */
FALCON_FUNC Window::get_icon_name( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* nam = gtk_window_get_icon_name( (GtkWindow*)_obj );
    if ( nam )
        vm->retval( UTF8String( nam ) );
    else
        vm->retnil();
}


/*#
    @method get_mnemonic_modifier GtkWindow
    @brief Returns the mnemonic modifier for this window.
    @return the modifier mask used to activate mnemonics on this window. (GdkModifierType)

    See set_mnemonic_modifier().
 */
FALCON_FUNC Window::get_mnemonic_modifier( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_window_get_mnemonic_modifier( (GtkWindow*)_obj ) );
}


/*#
    @method get_modal GtkWindow
    @brief Returns whether the window is modal.
    @return (boolean) true if the window is set to be modal and establishes a grab when shown

    See set_modal().
 */
FALCON_FUNC Window::get_modal( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_modal( (GtkWindow*)_obj ) );
}


/*#
    @method get_position GtkWindow
    @brief Gets the window position.
    @return an array [ root_x, root_y ]

    This function returns the position you need to pass to gtk_window_move() to
    keep window in its current position. This means that the meaning of the returned
    value varies with window gravity. See gtk_window_move() for more details.

    If you haven't changed the window gravity, its gravity will be GDK_GRAVITY_NORTH_WEST.
    This means that gtk_window_get_position() gets the position of the top-left corner of
    the window manager frame for the window. gtk_window_move() sets the position of this
    same top-left corner.

    gtk_window_get_position() is not 100% reliable because the X Window System does not
    specify a way to obtain the geometry of the decorations placed on a window by the
    window manager. Thus GTK+ is using a "best guess" that works with most window managers.

    Moreover, nearly all window managers are historically broken with respect to their
    handling of window gravity. So moving a window to its current position as returned by
    gtk_window_get_position() tends to result in moving the window slightly. Window managers
    are slowly getting better over time.

    If a window has gravity GDK_GRAVITY_STATIC the window manager frame is not relevant,
    and thus gtk_window_get_position() will always produce accurate results. However you
    can't use static gravity to do things like place a window in a corner of the screen,
    because static gravity ignores the window manager decorations.

    If you are saving and restoring your application's window positions, you should know
    that it's impossible for applications to do this without getting it somewhat wrong
    because applications do not have sufficient knowledge of window manager state.
    The Correct Mechanism is to support the session management protocol (see the
    "GnomeClient" object in the GNOME libraries for example) and allow the window
    manager to save your window sizes and positions.

 */
FALCON_FUNC Window::get_position( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint x, y;
    gtk_window_get_position( (GtkWindow*)_obj, &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( x );
    arr->append( y );
    vm->retval( arr );
}


/*#
    @method get_role GtkWindow
    @brief Returns the role of the window.
    @return the role of the window if set, or nil.

    See set_role() for further explanation.
 */
FALCON_FUNC Window::get_role( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* role = gtk_window_get_role( (GtkWindow*)_obj );
    if ( role )
        vm->retval( UTF8String( role ) );
    else
        vm->retnil();
}


/*#
    @method get_size GtkWindow
    @brief Obtains the current size of window.
    @brief an array [ width, height ]

    If window is not onscreen, it returns the size GTK+ will suggest to the window
    manager for the initial window size (but this is not reliably the same as the size
    the window manager will actually select). The size obtained by gtk_window_get_size()
    is the last size received in a GdkEventConfigure, that is, GTK+ uses its locally-stored
    size, rather than querying the X server for the size. As a result, if you call
    gtk_window_resize() then immediately call gtk_window_get_size(), the size won't have
    taken effect yet. After the window manager processes the resize request, GTK+ receives
    notification that the size has changed via a configure event, and the size of the
    window gets updated.

    Note 1: Nearly any use of this function creates a race condition, because the
    size of the window may change between the time that you get the size and the
    time that you perform some action assuming that size is the current size.
    To avoid race conditions, connect to "configure-event" on the window and adjust
    your size-dependent state to match the size delivered in the GdkEventConfigure.

    Note 2: The returned size does not include the size of the window manager
    decorations (aka the window frame or border). Those are not drawn by GTK+ and
    GTK+ has no reliable method of determining their size.

    Note 3: If you are getting a window size in order to position the window onscreen,
    there may be a better way. The preferred way is to simply set the window's semantic
    type with gtk_window_set_type_hint(), which allows the window manager to e.g.
    center dialogs. Also, if you set the transient parent of dialogs with
    gtk_window_set_transient_for() window managers will often center the dialog over
    its parent window. It's much preferred to let the window manager handle these things
    rather than doing it yourself, because all apps will behave consistently and according
    to user prefs if the window manager handles it. Also, the window manager can take the
    size of the window decorations/border into account, while your application cannot.

    In any case, if you insist on application-specified window positioning, there's
    still a better way than doing it yourself - gtk_window_set_position() will
    frequently handle the details for you.
 */
FALCON_FUNC Window::get_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint w, h;
    gtk_window_get_size( (GtkWindow*)_obj, &w, &h );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method get_title GtkWindow
    @brief Retrieves the title of the window.
    @return the title of the window, or nil if none has been set explicitely.
 */
FALCON_FUNC Window::get_title( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* t = gtk_window_get_title( (GtkWindow*)_obj );
    if ( t )
        vm->retval( UTF8String( t ) );
    else
        vm->retnil();
}


/*#
    @method get_transient_for GtkWindow
    @brief Fetches the transient parent for this window.
    @return the transient parent for this window, or NULL if no transient parent has been set.
 */
FALCON_FUNC Window::get_transient_for( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWindow* win = gtk_window_get_transient_for( (GtkWindow*)_obj );
    if ( win )
        vm->retval( new Gtk::Window( vm->findWKI( "GtkWindow" )->asClass(), win ) );
    else
        vm->retnil();
}


/*#
    @method get_type_hint GtkWindow
    @brief Gets the type hint for this window.
    @return the type hint for window. (GdkWindowTypeHint)
 */
FALCON_FUNC Window::get_type_hint( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_window_get_type_hint( (GtkWindow*)_obj ) );
}


/*#
    @method get_skip_taskbar_hint GtkWindow
    @brief Gets the value set by set_skip_taskbar_hint()
    @return true if window shouldn't be in taskbar (boolean)
 */
FALCON_FUNC Window::get_skip_taskbar_hint( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_skip_taskbar_hint( (GtkWindow*)_obj ) );
}


/*#
    @method get_skip_pager_hint GtkWindow
    @brief Gets the value set by set_skip_pager_hint().
    @return true if window shouldn't be in pager (boolean)
 */
FALCON_FUNC Window::get_skip_pager_hint( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_skip_pager_hint( (GtkWindow*)_obj ) );
}


/*#
    @method get_urgency_hint GtkWindow
    @brief Gets the value set by set_urgency_hint()
    @return true if window is urgent (boolean)
 */
FALCON_FUNC Window::get_urgency_hint( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_urgency_hint( (GtkWindow*)_obj ) );
}


/*#
    @method get_accept_focus GtkWindow
    @brief Gets the value set by set_accept_focus().
    @return true if window should receive the input focus (boolean)
 */
FALCON_FUNC Window::get_accept_focus( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_accept_focus( (GtkWindow*)_obj ) );
}


/*#
    @method get_focus_on_map GtkWindow
    @brief Gets the value set by gtk_window_set_focus_on_map().
    @return true if window should receive the input focus when mapped. (boolean)
 */
FALCON_FUNC Window::get_focus_on_map( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_get_focus_on_map( (GtkWindow*)_obj ) );
}


//FALCON_FUNC Window::get_group( VMARG );


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method get_window_type GtkWindow
    @brief Gets the type of the window.
    @return the type of the window (GtkWindowType)
 */
FALCON_FUNC Window::get_window_type( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_window_get_window_type( (GtkWindow*)_obj ) );
}
#endif


/*#
    @method move GtkWindow
    @brief Asks the window manager to move window to the given position.
    @param x X coordinate to move window to
    @param y Y coordinate to move window to

    Asks the window manager to move window to the given position. Window managers
    are free to ignore this; most window managers ignore requests for initial window
    positions (instead using a user-defined placement algorithm) and honor requests
    after the window has already been shown.

    Note: the position is the position of the gravity-determined reference point
    for the window. The gravity determines two things: first, the location of the
    reference point in root window coordinates; and second, which point on the window
    is positioned at the reference point.

    By default the gravity is GDK_GRAVITY_NORTH_WEST, so the reference point is
    simply the x, y supplied to gtk_window_move(). The top-left corner of the window
    decorations (aka window frame or border) will be placed at x, y. Therefore, to
    position a window at the top left of the screen, you want to use the default
    gravity (which is GDK_GRAVITY_NORTH_WEST) and move the window to 0,0.

    To position a window at the bottom right corner of the screen, you would set
    GDK_GRAVITY_SOUTH_EAST, which means that the reference point is at x + the window
    width and y + the window height, and the bottom-right corner of the window border
    will be placed at that reference point. So, to place a window in the bottom right
    corner you would first set gravity to south east, then write: gtk_window_move
    (window, gdk_screen_width() - window_width, gdk_screen_height() - window_height)
    (note that this example does not take multi-head scenarios into account).

    The Extended Window Manager Hints specification at
    http://www.freedesktop.org/Standards/wm-spec has a nice table of gravities in
    the "implementation notes" section.

    The gtk_window_get_position() documentation may also be relevant.
 */
FALCON_FUNC Window::move( VMARG )
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
    gtk_window_move( (GtkWindow*)_obj, i_x->asInteger(), i_y->asInteger() );
}


/*#
    @method parse_geometry GtkWindow
    @brief Parses a standard X Window System geometry string
    @param geometry geometry string
    @return true if string was parsed successfully (boolean)

    Parses a standard X Window System geometry string - see the manual page for X
    (type 'man X') for details on this. gtk_window_parse_geometry() does work on all
    GTK+ ports including Win32 but is primarily intended for an X environment.

    If either a size or a position can be extracted from the geometry string,
    gtk_window_parse_geometry() returns TRUE and calls gtk_window_set_default_size()
    and/or gtk_window_move() to resize/move the window.

    If gtk_window_parse_geometry() returns TRUE, it will also set the GDK_HINT_USER_POS
    and/or GDK_HINT_USER_SIZE hints indicating to the window manager that the size/position
    of the window was user-specified. This causes most window managers to honor the geometry.

    Note that for gtk_window_parse_geometry() to work as expected, it has to be called
    when the window has its "final" size, i.e. after calling gtk_widget_show_all() on
    the contents and gtk_window_set_geometry_hints() on the window.
 */
FALCON_FUNC Window::parse_geometry( VMARG )
{
    Item* i_geom = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_geom || !i_geom->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString geom( i_geom->asString() );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_window_parse_geometry( (GtkWindow*)_obj, geom.c_str() ) );
}


/*#
    @method reshow_initial_size GtkWindow
    @breif Hides window, then reshows it, resetting the default size and position of the window.

    Used by GUI builders only.
 */
FALCON_FUNC Window::reshow_with_initial_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_window_reshow_with_initial_size( (GtkWindow*)_obj );
}


/*#
    @method resize GtkWindow
    @brief Resizes the window as if the user had done so, obeying geometry constraints.
    @param width width in pixels to resize the window to
    @param height height in pixels to resize the window to

    The default geometry constraint is that windows may not be smaller than their
    size request; to override this constraint, call gtk_widget_set_size_request()
    to set the window's request to a smaller value.

    If gtk_window_resize() is called before showing a window for the first time,
    it overrides any default size set with gtk_window_set_default_size().

    Windows may not be resized smaller than 1 by 1 pixels.
 */
FALCON_FUNC Window::resize( VMARG )
{
    Item* i_w = vm->param( 0 );
    Item* i_h = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || !i_w->isInteger()
        || !i_h || !i_h->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_resize( (GtkWindow*)_obj, i_w->asInteger(), i_h->asInteger() );
}


//FALCON_FUNC Window::set_default_icon_list( VMARG );

//FALCON_FUNC Window::set_default_icon( VMARG );

//FALCON_FUNC Window::set_default_icon_from_file( VMARG );


/*#
    @method set_default_icon_name GtkWindow
    @brief Sets an icon to be used as fallback.
    @param name the name of the themed icon

    Sets an icon to be used as fallback for windows that haven't had set_icon_list()
    called on them from a named themed icon, see set_icon_name().
 */
FALCON_FUNC Window::set_default_icon_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !i_name->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString name( i_name->asString() );
    gtk_window_set_default_icon_name( name.c_str() );
}


//FALCON_FUNC Window::set_icon( VMARG );

//FALCON_FUNC Window::set_icon_list( VMARG );

//FALCON_FUNC Window::set_icon_from_file( VMARG );


/*#
    @method set_icon_name GtkWindow
    @brief Sets the icon for the window from a named themed icon.
    @param the name of the themed icon (or nil)

    See the docs for GtkIconTheme for more details.

    Note that this has nothing to do with the WM_ICON_NAME property which is
    mentioned in the ICCCM.
 */
FALCON_FUNC Window::set_icon_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !( i_name->isNil() || i_name->isString() ) )
        throw_inv_params( "[S]" );
#endif
    MYSELF;
    GET_OBJ( self );
    if ( i_name->isString() )
    {
        AutoCString name( i_name->asString() );
        gtk_window_set_icon_name( (GtkWindow*)_obj, name.c_str() );
    }
    else
        gtk_window_set_icon_name( (GtkWindow*)_obj, NULL );
}


/*#
    @method set_auto_startup_notification GtkWindow
    @brief Sets the window startup notification.
    @param setting true to automatically do startup notification (boolean)

    By default, after showing the first GtkWindow, GTK+ calls gdk_notify_startup_complete().
    Call this function to disable the automatic startup notification. You might do
    this if your first window is a splash screen, and you want to delay notification
    until after your real main window has been shown, for example.

    In that example, you would disable startup notification temporarily, show your
    splash screen, then re-enable it so that showing the main window would automatically
    result in notification.
 */
FALCON_FUNC Window::set_auto_startup_notification( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_window_set_auto_startup_notification( (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_opacity GtkWindow
    @brief Fetches the requested opacity for this window.
    @return the requested opacity for this window. (float)

    See set_opacity().
 */
FALCON_FUNC Window::get_opacity( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_window_get_opacity( (GtkWindow*)_obj ) );
}


/*#
    @method set_opacity GtkWindow
    @brief Sets the window opacity.
    @param opacity desired opacity, between 0 and 1 (float)

    Request the windowing system to make window partially transparent, with opacity
    0 being fully transparent and 1 fully opaque. (Values of the opacity parameter
    are clamped to the [0,1] range.) On X11 this has any effect only on X screens
    with a compositing manager running. See gtk_widget_is_composited(). On Windows
    it should work always.

    Note that setting a window's opacity after the window has been shown causes it
    to flicker once on Windows.
 */
FALCON_FUNC Window::set_opacity( VMARG )
{
    Item* i_opac = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_opac || !i_opac->isOrdinal() )
        throw_inv_params( "N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_window_set_opacity( (GtkWindow*)_obj, i_opac->forceNumeric() );
}


//FALCON_FUNC Window::get_mnemonics_visible( VMARG );

//FALCON_FUNC Window::set_mnemonics_visible( VMARG );


} // Gtk
} // Falcon
