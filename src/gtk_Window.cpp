/**
 *  \file gtk_Window.cpp
 */

#include "gtk_Window.hpp"

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
    //{ "get_gravity",        &Window::set_gravity },
    //{ "set_position",        &Window::set_gravity },
    //{ "set_transient_for",        &Window::set_gravity },
    //{ "set_destroy_with_parent",        &Window::set_gravity },
    //{ "set_screen",        &Window::set_gravity },
    //{ "get_screen",        &Window::set_gravity },
    //{ "is_active",        &Window::set_gravity },
    //{ "has_toplevel_focus",        &Window::set_gravity },
    //{ "list_toplevels",        &Window::set_gravity },
    //{ "add_mnemonic",        &Window::set_gravity },
    //{ "remove_mnemonic",        &Window::set_gravity },
    //{ "mnemonic_activate",        &Window::set_gravity },
    //{ "activate_key",        &Window::set_gravity },
    //{ "propagate_key_event",        &Window::set_gravity },
    //{ "get_focus",        &Window::set_gravity },
    //{ "set_focus",        &Window::set_gravity },
    //{ "get_default_widget",        &Window::set_gravity },
    //{ "set_default",        &Window::set_gravity },
    //{ "present",        &Window::set_gravity },
    //{ "present_with_time",        &Window::set_gravity },
    //{ "iconify",        &Window::set_gravity },
    //{ "deiconify",        &Window::set_gravity },
    //{ "stick",        &Window::set_gravity },
    //{ "unstick",        &Window::set_gravity },
    //{ "maximize",        &Window::set_gravity },
    //{ "unmaximize",        &Window::set_gravity },
    //{ "fullscreen",        &Window::set_gravity },
    //{ "unfullscreen",        &Window::set_gravity },
    //{ "set_keep_above",        &Window::set_gravity },
    //{ "set_keep_below",        &Window::set_gravity },
    //{ "begin_resize_drag",        &Window::set_gravity },
    //{ "begin_move_drag",        &Window::set_gravity },
    //{ "set_decorated",        &Window::set_gravity },
    //{ "set_deletable",        &Window::set_gravity },
    //{ "set_frame_dimensions",        &Window::set_gravity },
    //{ "set_has_frame",        &Window::set_gravity },
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


//FALCON_FUNC Window::get_gravity( VMARG );


//FALCON_FUNC Window::set_position( VMARG );

//FALCON_FUNC Window::set_transient_for( VMARG );

//FALCON_FUNC Window::set_destroy_with_parent( VMARG );

//FALCON_FUNC Window::set_screen( VMARG );

//FALCON_FUNC Window::get_screen( VMARG );

//FALCON_FUNC Window::is_active( VMARG );

//FALCON_FUNC Window::has_toplevel_focus( VMARG );

//FALCON_FUNC Window::list_toplevels( VMARG );

//FALCON_FUNC Window::add_mnemonic( VMARG );

//FALCON_FUNC Window::remove_mnemonic( VMARG );

//FALCON_FUNC Window::mnemonic_activate( VMARG );

//FALCON_FUNC Window::activate_key( VMARG );

//FALCON_FUNC Window::propagate_key_event( VMARG );

//FALCON_FUNC Window::get_focus( VMARG );

//FALCON_FUNC Window::set_focus( VMARG );

//FALCON_FUNC Window::get_default_widget( VMARG );

//FALCON_FUNC Window::set_default( VMARG );

//FALCON_FUNC Window::present( VMARG );

//FALCON_FUNC Window::present_with_time( VMARG );

//FALCON_FUNC Window::iconify( VMARG );

//FALCON_FUNC Window::deiconify( VMARG );

//FALCON_FUNC Window::stick( VMARG );

//FALCON_FUNC Window::unstick( VMARG );

//FALCON_FUNC Window::maximize( VMARG );

//FALCON_FUNC Window::unmaximize( VMARG );

//FALCON_FUNC Window::fullscreen( VMARG );

//FALCON_FUNC Window::unfullscreen( VMARG );

//FALCON_FUNC Window::set_keep_above( VMARG );

//FALCON_FUNC Window::set_keep_below( VMARG );

//FALCON_FUNC Window::begin_resize_drag( VMARG );

//FALCON_FUNC Window::begin_move_drag( VMARG );

//FALCON_FUNC Window::set_decorated( VMARG );

//FALCON_FUNC Window::set_deletable( VMARG );

//FALCON_FUNC Window::set_frame_dimensions( VMARG );

//FALCON_FUNC Window::set_has_frame( VMARG );

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
