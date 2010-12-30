/**
 *  \file gdk_Display.cpp
 */

#include "gdk_Display.hpp"

// #include "gdk_Device.hpp"
#include "gdk_Screen.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Display::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Display = mod->addClass( "GdkDisplay", &Display::open );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Display->getClassDef()->addInheritance( in );

    c_Display->setWKS( true );
    c_Display->getClassDef()->factory( &Display::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_closed",          Display::signal_closed },
    { "open",                   Display::open },
    { "get_default",            Display::get_default },
    { "get_name",               Display::get_name },
    { "get_n_screens",          Display::get_n_screens },
    { "get_screen",             Display::get_screen },
    { "get_default_screen",     Display::get_default_screen },
    { "pointer_ungrab",         Display::pointer_ungrab },
    { "keyboard_ungrab",        Display::keyboard_ungrab },
    { "pointer_is_grabbed",     Display::pointer_is_grabbed },
#if 0 // todo
    { "device_is_grabbed",      Display::device_is_grabbed },
#endif
    { "beep",                   Display::beep },
    { "sync",                   Display::sync },
    { "flush",                  Display::flush },
    { "close",                  Display::close },
#if 0 // todo
    { "list_devices",           Display::list_devices },
    { "get_event",              Display::get_event },
    { "peek_event",             Display::peek_event },
    { "put_event",              Display::put_event },
    { "add_client_message_filter",Display::add_client_message_filter },
    { "set_double_click_time",  Display::set_double_click_time },
    { "set_double_click_distance",Display::set_double_click_distance },
    { "get_pointer",            Display::get_pointer },
    { "get_window_at_pointer",  Display::get_window_at_pointer },
    { "set_pointer_hooks",      Display::set_pointer_hooks },
    { "warp_pointer",           Display::warp_pointer },
    { "supports_cursor_color",  Display::supports_cursor_color },
    { "supports_cursor_alpha",  Display::supports_cursor_alpha },
    { "get_default_cursor_size",Display::get_default_cursor_size },
    { "get_maximal_cursor_size",Display::get_maximal_cursor_size },
    { "get_default_group",      Display::get_default_group },
    { "supports_selection_notification",Display::supports_selection_notification },
    { "request_selection_notification",Display::request_selection_notification },
    { "supports_clipboard_persistence",Display::supports_clipboard_persistence },
    { "store_clipboard",        Display::store_clipboard },
    { "supports_shapes",        Display::supports_shapes },
    { "supports_input_shapes",  Display::supports_input_shapes },
    { "supports_composite",     Display::supports_composite },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Display, meth->name, meth->cb );
}


Display::Display( const Falcon::CoreClass* gen, const GdkDisplay* bmap )
    :
    Gtk::CoreGObject( gen, (GObject*) bmap )
{}


Falcon::CoreObject* Display::factory( const Falcon::CoreClass* gen, void* bmap, bool )
{
    return new Display( gen, (GdkDisplay*) bmap );
}


/*#
    @class GdkDisplay
    @brief An opaque structure representing an offscreen drawable of depth 1.
    @param display_name the name of the display to open

    GdkDisplay objects purpose are two fold:

    - To grab/ungrab keyboard focus and mouse pointer
    - To manage and provide information about the GdkScreen(s) available for this GdkDisplay

    GdkDisplay objects are the GDK representation of the X Display which can be
    described as a workstation consisting of a keyboard a pointing device (such
    as a mouse) and one or more screens. It is used to open and keep track of
    various GdkScreen objects currently instanciated by the application. It is
    also used to grab and release the keyboard and the mouse pointer.
 */
FALCON_FUNC Display::open( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !i_name->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString name( i_name->asString() );
    MYSELF;
    GdkDisplay* display = gdk_display_open( name.c_str() );
    if ( display )
        self->setObject( (GObject*) display );
    else
        throw_inv_params( "Display could not be opened" ); // todo
}


/*#
    @method signal_closed GdkDisplay
    @brief The closed signal is emitted when the connection to the windowing system for display is closed.
 */
FALCON_FUNC Display::signal_closed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "closed", (void*) &Display::on_closed, vm );
}


void Display::on_closed( GdkDisplay* obj, gboolean is_error, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "closed", false );

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
                || !it.asObject()->getMethod( "on_closed", it ) )
            {
                printf(
                "[GdkDisplay::on_closed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( (bool) is_error );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method signal_opened GdkDisplay
    @brief The opened signal is emitted when the connection to the windowing system for display is opened.
 */
FALCON_FUNC Display::signal_opened( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "opened", (void*) &Display::on_opened, vm );
}


void Display::on_opened( GdkDisplay* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "opened", "on_opened", (VMachine*)_vm );
}


/*#
    @method get_default GdkDisplay
    @brief Gets the default GdkDisplay.
    @return a GdkDisplay, or NULL if there is no default display.

    This is a convenience function for gdk_display_manager_get_default_display (gdk_display_manager_get()).
 */
FALCON_FUNC Display::get_default( VMARG )
{
    NO_ARGS
    GdkDisplay* display = gdk_display_get_default();
    if ( display )
        vm->retval( new Gdk::Display( vm->findWKI( "GdkDisplay" )->asClass(), display ) );
    else
        vm->retnil();
}


/*#
    @method get_name GdkDisplay
    @brief Gets the name of the display.
    @return a string representing the display name.
 */
FALCON_FUNC Display::get_name( VMARG )
{
    NO_ARGS
    vm->retval( UTF8String( gdk_display_get_name( GET_DISPLAY( vm->self() ) ) ) );
}


/*#
    @method get_n_screens GdkDisplay
    @brief Gets the number of screen managed by the display.
    @return number of screens.
 */
FALCON_FUNC Display::get_n_screens( VMARG )
{
    NO_ARGS
    vm->retval( gdk_display_get_n_screens( GET_DISPLAY( vm->self() ) ) );
}


/*#
    @method get_screen GdkDisplay
    @brief Returns a screen object for one of the screens of the display.
    @param screen_num the screen number
    @return the GdkScreen object
 */
FALCON_FUNC Display::get_screen( VMARG )
{
    Item* i_n = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_n || !i_n->isInteger() )
        throw_inv_params( "I" );
#endif
    vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(),
                                 gdk_display_get_screen( GET_DISPLAY( vm->self() ),
                                                         i_n->asInteger() ) ) );
}


/*#
    @method get_default_screen GdkDisplay
    @brief Get the default GdkScreen for display.
    @return the default GdkScreen object for display
 */
FALCON_FUNC Display::get_default_screen( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(),
                gdk_display_get_default_screen( GET_DISPLAY( vm->self() ) ) ) );
}


/*#
    @method pointer_ungrab GdkDisplay
    @brief Release any pointer grab.
    @param time a timestamp (e.g. GDK_CURRENT_TIME).

    Warning: gdk_display_pointer_ungrab has been deprecated since version 3.0
    and should not be used in newly-written code. Use gdk_device_ungrab(),
    together with gdk_device_grab()  instead
 */
FALCON_FUNC Display::pointer_ungrab( VMARG )
{
    Item* i_t = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_t || !i_t->isInteger() )
        throw_inv_params( "I" );
#endif
    gdk_display_pointer_ungrab( GET_DISPLAY( vm->self() ), i_t->asInteger() );
}


/*#
    @method keyboard_ungrab GdkDisplay
    @brief Release any keyboard grab
    @param time a timestap (e.g GDK_CURRENT_TIME).

    Warning: gdk_display_keyboard_ungrab has been deprecated since version 3.0
    and should not be used in newly-written code. Use gdk_device_ungrab(),
    together with gdk_device_grab()  instead.
 */
FALCON_FUNC Display::keyboard_ungrab( VMARG )
{
    Item* i_t = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_t || !i_t->isInteger() )
        throw_inv_params( "I" );
#endif
    gdk_display_keyboard_ungrab( GET_DISPLAY( vm->self() ), i_t->asInteger() );
}


/*#
    @method pointer_is_grabbed GdkDisplay
    @brief Test if the pointer is grabbed.
    @return TRUE if an active X pointer grab is in effect

    Warning: gdk_display_pointer_is_grabbed has been deprecated since version 3.0
    and should not be used in newly-written code. Use gdk_display_device_is_grabbed()
    instead.
 */
FALCON_FUNC Display::pointer_is_grabbed( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gdk_display_pointer_is_grabbed( GET_DISPLAY( vm->self() ) ) );
}

#if 0 // todo
/*#
    @method device_is_grabbed GdkDisplay
    @brief Returns TRUE if there is an ongoing grab on device for display.
    @param device a GdkDevice
    @return TRUE if there is a grab in effect for device.

    @note This method is currently **NOT** implemented.
 */
FALCON_FUNC Display::device_is_grabbed( VMARG )
{

}
#endif

/*#
    @method beep GdkDisplay
    @brief Emits a short beep on display
 */
FALCON_FUNC Display::beep( VMARG )
{
    NO_ARGS
    gdk_display_beep( GET_DISPLAY( vm->self() ) );
}


/*#
    @method sync GdkDisplay
    @brief Flushes any requests queued for the windowing system and waits until all requests have been handled.

    This is often used for making sure that the display is synchronized with the
    current state of the program. Calling gdk_display_sync() before
    gdk_error_trap_pop() makes sure that any errors generated from earlier
    requests are handled before the error trap is removed.

    This is most useful for X11. On windowing systems where requests are handled
    synchronously, this function will do nothing.
 */
FALCON_FUNC Display::sync( VMARG )
{
    NO_ARGS
    gdk_display_sync( GET_DISPLAY( vm->self() ) );
}


/*#
    @method flush GdkDisplay
    @brief Flushes any requests queued for the windowing system.

    This happens automatically when the main loop blocks waiting for new events,
    but if your application is drawing without returning control to the main loop,
    you may need to call this function explicitely. A common case where this
    function needs to be called is when an application is executing drawing
    commands from a thread other than the thread where the main loop is running.

    This is most useful for X11. On windowing systems where requests are handled
    synchronously, this function will do nothing.
 */
FALCON_FUNC Display::flush( VMARG )
{
    NO_ARGS
    gdk_display_flush( GET_DISPLAY( vm->self() ) );
}


/*#
    @method close GdkDisplay
    @brief Closes the connection to the windowing system for the given display, and cleans up associated resources.
 */
FALCON_FUNC Display::close( VMARG )
{
    NO_ARGS
    gdk_display_close( GET_DISPLAY( vm->self() ) );
}


#if 0 // todo
FALCON_FUNC Display::list_devices( VMARG );


FALCON_FUNC Display::get_event( VMARG );
FALCON_FUNC Display::peek_event( VMARG );
FALCON_FUNC Display::put_event( VMARG );
FALCON_FUNC Display::add_client_message_filter( VMARG );
FALCON_FUNC Display::set_double_click_time( VMARG );
FALCON_FUNC Display::set_double_click_distance( VMARG );
FALCON_FUNC Display::get_pointer( VMARG );
FALCON_FUNC Display::get_window_at_pointer( VMARG );
FALCON_FUNC Display::set_pointer_hooks( VMARG );
FALCON_FUNC Display::warp_pointer( VMARG );
FALCON_FUNC Display::supports_cursor_color( VMARG );
FALCON_FUNC Display::supports_cursor_alpha( VMARG );
FALCON_FUNC Display::get_default_cursor_size( VMARG );
FALCON_FUNC Display::get_maximal_cursor_size( VMARG );
FALCON_FUNC Display::get_default_group( VMARG );
FALCON_FUNC Display::supports_selection_notification( VMARG );
FALCON_FUNC Display::request_selection_notification( VMARG );
FALCON_FUNC Display::supports_clipboard_persistence( VMARG );
FALCON_FUNC Display::store_clipboard( VMARG );
FALCON_FUNC Display::supports_shapes( VMARG );
FALCON_FUNC Display::supports_input_shapes( VMARG );
FALCON_FUNC Display::supports_composite( VMARG );
#endif
} // Gdk
} // Falcon
