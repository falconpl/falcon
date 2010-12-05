/**
 *  \file gdk_Event.cpp
 */

#include "gdk_Event.hpp"

#include "gdk_EventButton.hpp"
//#include "gdk_Screen.hpp"

#undef MYSELF
#define MYSELF Gdk::Event* self = Falcon::dyncast<Gdk::Event*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Event::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Event = mod->addClass( "GdkEvent", &Event::init );

    c_Event->setWKS( true );
    c_Event->getClassDef()->factory( &Event::factory );

    mod->addClassProperty( c_Event, "type" );
    //mod->addClassProperty( c_Event, "window" );
    mod->addClassProperty( c_Event, "send_event" );

    mod->addClassMethod( c_Event, "get_real_event", &Event::get_real_event );
    mod->addClassMethod( c_Event, "events_pending", &Event::events_pending );
    mod->addClassMethod( c_Event, "peek",           &Event::peek );
    mod->addClassMethod( c_Event, "get",            &Event::get );
#if 0 // deprecated
    mod->addClassMethod( c_Event, "get_graphics_expose",&Event::get_graphics_expose );
#endif
    mod->addClassMethod( c_Event, "put",            &Event::put );
    mod->addClassMethod( c_Event, "copy",           &Event::copy );
    mod->addClassMethod( c_Event, "get_state",      &Event::get_state );
    mod->addClassMethod( c_Event, "get_axis",       &Event::get_axis );
    mod->addClassMethod( c_Event, "get_coords",     &Event::get_coords );
    mod->addClassMethod( c_Event, "get_root_coords",&Event::get_root_coords );
    mod->addClassMethod( c_Event, "get_show_events",&Event::get_show_events );
    mod->addClassMethod( c_Event, "set_show_events",&Event::set_show_events );
#if 0 // todo
    mod->addClassMethod( c_Event, "set_screen",     &Event::set_screen );
    mod->addClassMethod( c_Event, "get_screen",     &Event::get_screen );
#endif
    // todo: some remaining functions.

    // related constants
    mod->addConstant( "GDK_CURRENT_TIME",       (int64) GDK_CURRENT_TIME );
    mod->addConstant( "GDK_PRIORITY_EVENTS",    (int64) GDK_PRIORITY_EVENTS );
    mod->addConstant( "GDK_PRIORITY_REDRAW",    (int64) GDK_PRIORITY_REDRAW );
}


Event::Event( const Falcon::CoreClass* gen, const GdkEvent* ev, const bool transfer )
    :
    Gtk::VoidObject( gen, ev )
{
    if ( m_obj && !transfer )
        m_obj = gdk_event_copy( (GdkEvent*) m_obj );
}


Event::Event( const Event& other )
    :
    Gtk::VoidObject( other )
{
    if ( m_obj )
        m_obj = gdk_event_copy( (GdkEvent*) m_obj );
}


Event::~Event()
{
    if ( m_obj )
        gdk_event_free( (GdkEvent*) m_obj );
}


void Event::setObject( const void* ev, const bool transfer )
{
    VoidObject::setObject( ev );
    if ( !transfer )
        m_obj = gdk_event_copy( (GdkEvent*) ev );
}


bool Event::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkEvent* m_event = (GdkEvent*) m_obj;

    if ( s == "type" )
        it = (int64) m_event->type;
#if 0 // todo
    else
    if ( s == "window" )
        it = ;
#endif
    else
    if ( s == "send_event" )
        it = (bool) ((GdkEventAny*)m_event)->send_event;
    else
        return defaultProperty( s, it );
    return true;
}


bool Event::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* Event::factory( const Falcon::CoreClass* gen, void* ev, bool )
{
    return new Event( gen, (GdkEvent*) ev );
}


/*#
    @class GdkEvent
    @brief Functions for handling events from the window system
    @param type a GdkEventType

    @prop type the type of the event (GdkEventType).
    @prop window TODO the window which received the event.
    @prop send_event TRUE if the event was sent explicitly (e.g. using XSendEvent).

    Creates a new event of the given type. All fields are set to 0.
 */
FALCON_FUNC Event::init( VMARG )
{
    MYSELF;
    if ( self->getObject() )
        return;

    Item* i_type = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_type || !i_type->isInteger() )
        throw_inv_params( "GdkEventType" );
#endif
    self->setObject( (GdkEvent*) gdk_event_new( (GdkEventType) i_type->asInteger() ),
                    true );
}


/*#
    @method get_real_event GdkEvent
    @brief Get a copy of the event cast to its real derived-type.
    @return one of the Gdk event types.
    @note This is Falcon specific.

    In classes already derived from GdkEvent, this is of course not of much use.
 */
FALCON_FUNC Event::get_real_event( VMARG )
{
    NO_ARGS
    MYSELF;
    switch ( self->getObject()->type )
    {
#if 0
    case GDK_NOTHING:
    case GDK_DELETE:
    case GDK_DESTROY:
    case GDK_EXPOSE:
    case GDK_MOTION_NOTIFY:
#endif
    case GDK_BUTTON_PRESS:
    case GDK_2BUTTON_PRESS:
    case GDK_3BUTTON_PRESS:
    case GDK_BUTTON_RELEASE:
    {
        vm->retval( new Gdk::EventButton( vm->findWKI( "GdkEventButton" )->asClass(),
                                          (GdkEventButton*) self->getObject() ) );
        return;
    }
#if 0
    case GDK_KEY_PRESS:
    case GDK_KEY_RELEASE:
    case GDK_ENTER_NOTIFY:
    case GDK_LEAVE_NOTIFY:
    case GDK_FOCUS_CHANGE:
    case GDK_CONFIGURE:
    case GDK_MAP:
    case GDK_UNMAP:
    case GDK_PROPERTY_NOTIFY:
    case GDK_SELECTION_CLEAR:
    case GDK_SELECTION_REQUEST:
    case GDK_SELECTION_NOTIFY:
    case GDK_PROXIMITY_IN:
    case GDK_PROXIMITY_OUT:
    case GDK_DRAG_ENTER:
    case GDK_DRAG_LEAVE:
    case GDK_DRAG_MOTION:
    case GDK_DRAG_STATUS:
    case GDK_DROP_START:
    case GDK_DROP_FINISHED:
    case GDK_CLIENT_EVENT:
    case GDK_VISIBILITY_NOTIFY:
    case GDK_NO_EXPOSE:
    case GDK_SCROLL:
    case GDK_WINDOW_STATE:
    case GDK_SETTING:
    case GDK_OWNER_CHANGE:
    case GDK_GRAB_BROKEN:
#if GTK_CHECK_VERSION( 2, 14, 0 )
    case GDK_DAMAGE:
#endif
#if 0 // not reached
    case GDK_EVENT_LAST:
#endif
#endif
    default:
        return; // not reached
    }
}


/*#
    @method events_pending GdkEvent
    @brief Checks if any events are ready to be processed for any display.
    @return TRUE if any events are pending.
 */
FALCON_FUNC Event::events_pending( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gdk_events_pending() );
}


/*#
    @method peek GdkEvent
    @brief If there is an event waiting in the event queue of some open display, returns it.
    @return the first GdkEvent on some event queue, or NULL if no events are in any queues.

    See gdk_display_peek_event().
 */
FALCON_FUNC Event::peek( VMARG )
{
    NO_ARGS
    GdkEvent* ev = gdk_event_peek();
    if ( ev )
        vm->retval( new Gdk::Event( vm->findWKI( "GdkEvent" )->asClass(), ev, true ) );
    else
        vm->retnil();
}


/*#
    @method get GdkEvent
    @brief Checks all open displays for a GdkEvent to process, to be processed on, fetching events from the windowing system if necessary.
    @return the next GdkEvent to be processed, or NULL if no events are pending.

    See gdk_display_get_event().
 */
FALCON_FUNC Event::get( VMARG )
{
    NO_ARGS
    GdkEvent* ev = gdk_event_get();
    if ( ev )
        vm->retval( new Gdk::Event( vm->findWKI( "GdkEvent" )->asClass(), ev, true ) );
    else
        vm->retnil();
}


#if 0 // deprecated
FALCON_FUNC Event::get_graphics_expose( VMARG );
#endif


/*#
    @method put GdkEvent
    @brief Appends a copy of the given event onto the front of the event queue for event->any.window's display, or the default event queue if event->any.window is NULL.

    See gdk_display_put_event().
 */
FALCON_FUNC Event::put( VMARG )
{
    NO_ARGS
    gdk_event_put( GET_EVENT( vm->self() ) );
}


/*#
    @method copy GdkEvent
    @brief Copies a GdkEvent, copying or incrementing the reference count of the resources associated with it (e.g. GdkWindow's and strings).
    @return a copy of the event.
 */
FALCON_FUNC Event::copy( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Event( vm->findWKI( "GdkEvent" )->asClass(),
                                GET_EVENT( vm->self() ) ) );
}


/*#
    @method get_time GdkEvent
    @brief Returns the time stamp from event, if there is one; otherwise returns GDK_CURRENT_TIME.
    @return time stamp field from event
 */
FALCON_FUNC Event::get_time( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gdk_event_get_time( GET_EVENT( vm->self() ) ) );
}


/*#
    @method get_state GdkEvent
    @brief If the event contains a "state" field, returns it. Otherwise returns an empty state (0).
    @return the state (GdkModifierType).
 */
FALCON_FUNC Event::get_state( VMARG )
{
    NO_ARGS
    GdkModifierType state;
    gdk_event_get_state( GET_EVENT( vm->self() ), &state );
    vm->retval( (int64) state );
}


/*#
    @method get_axis GdkEvent
    @brief Extract the axis value for a particular axis use from an event structure.
    @param axis_use the axis use to look for (GdkAxisUse).
    @return the value found or nil if the specified axis was not found.
 */
FALCON_FUNC Event::get_axis( VMARG )
{
    Item* i_ax = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ax || !i_ax->isInteger() )
        throw_inv_params( "GdkAxisUse" );
#endif
    gdouble value;
    gboolean ret = gdk_event_get_axis( GET_EVENT( vm->self() ),
                                       (GdkAxisUse) i_ax->asInteger(), &value );
    if ( ret )
        vm->retval( (numeric) value );
    else
        vm->retnil();
}


/*#
    @method get_coords GdkEvent
    @brief Extract the event window relative x/y coordinates from an event.
    @return an array ( window x coordinate, window y coordinate ), or nil if the event did not deliver event window coordinates.
 */
FALCON_FUNC Event::get_coords( VMARG )
{
    NO_ARGS
    gdouble x, y;
    gboolean ret = gdk_event_get_coords( GET_EVENT( vm->self() ), &x, &y );
    if ( ret )
    {
        CoreArray* arr = new CoreArray( 2 );
        arr->append( (numeric) x );
        arr->append( (numeric) y );
        vm->retval( arr );
    }
    else
        vm->retnil();
}


/*#
    @method get_root_coords GdkEvent
    @brief Extract the root window relative x/y coordinates from an event.
    @return an array ( window x coordinate, window y coordinate ), or nil if the event did not deliver root window coordinates
 */
FALCON_FUNC Event::get_root_coords( VMARG )
{
    NO_ARGS
    gdouble x, y;
    gboolean ret = gdk_event_get_root_coords( GET_EVENT( vm->self() ), &x, &y );
    if ( ret )
    {
        CoreArray* arr = new CoreArray( 2 );
        arr->append( (numeric) x );
        arr->append( (numeric) y );
        vm->retval( arr );
    }
    else
        vm->retnil();
}


/*#
    @method get_show_events GdkEvent
    @brief Gets whether event debugging output is enabled.
    @return TRUE if event debugging output is enabled.
 */
FALCON_FUNC Event::get_show_events( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gdk_get_show_events() );
}


/*#
    @method set_show_events GdkEvent
    @brief Sets whether a trace of received events is output.
    @param show_events TRUE to output event debugging information.

    Note that GTK+ must be compiled with debugging (that is, configured using
    the --enable-debug option) to use this option.
 */
FALCON_FUNC Event::set_show_events( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gdk_set_show_events( (gboolean) i_bool->asBoolean() );
}


#if 0 // todo
FALCON_FUNC Event::set_screen( VMARG );
FALCON_FUNC Event::get_screen( VMARG );
#endif


} // Gdk
} // Falcon
