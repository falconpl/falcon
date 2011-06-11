/**
 *  \file gdk_EventButton.cpp
 */

#include "gdk_EventButton.hpp"

#undef MYSELF
#define MYSELF Gdk::EventButton* self = Falcon::dyncast<Gdk::EventButton*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void EventButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_EventButton = mod->addClass( "GdkEventButton", &EventButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GdkEvent" ) );
    c_EventButton->getClassDef()->addInheritance( in );

    c_EventButton->setWKS( true );
    c_EventButton->getClassDef()->factory( &EventButton::factory );

    mod->addClassProperty( c_EventButton, "time" );
    mod->addClassProperty( c_EventButton, "x" );
    mod->addClassProperty( c_EventButton, "y" );
    //mod->addClassProperty( c_EventButton, "axes" );
    mod->addClassProperty( c_EventButton, "state" );
    mod->addClassProperty( c_EventButton, "button" );
    //mod->addClassProperty( c_EventButton, "device" );
    mod->addClassProperty( c_EventButton, "x_root" );
    mod->addClassProperty( c_EventButton, "y_root" );
}


EventButton::EventButton( const Falcon::CoreClass* gen,
                          const GdkEventButton* ev, const bool transfer )
    :
    Gdk::Event( gen, (GdkEvent*) ev, transfer )
{}


EventButton::EventButton( const EventButton& other )
    :
    Gdk::Event( other )
{}


EventButton::~EventButton()
{}


void EventButton::setObject( const void* ev, const bool transfer )
{
    Gdk::Event::setObject( ev, transfer );
}


bool EventButton::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkEventButton* m_event = (GdkEventButton*) m_obj;

    if ( s == "time" )
        it = ((GdkEventButton*)m_event)->time;
    else
    if ( s == "x" )
        it = ((GdkEventButton*)m_event)->x;
    else
    if ( s == "y" )
        it = ((GdkEventButton*)m_event)->y;
#if 0 // todo
    else
    if ( s == "axes" )
        it = m_event->axis;
#endif
    else
    if ( s == "state" )
        it = ((GdkEventButton*)m_event)->state;
    else
    if ( s == "button" )
        it = ((GdkEventButton*)m_event)->button;
    else
    if ( s == "x_root" )
        it = ((GdkEventButton*)m_event)->x_root;
    else
    if ( s == "y_root" )
        it = ((GdkEventButton*)m_event)->y_root;
    else
        return Gdk::Event::getProperty( s, it );
    return true;
}


bool EventButton::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* EventButton::factory( const Falcon::CoreClass* gen, void* ev, bool )
{
    return new EventButton( gen, (GdkEventButton*) ev );
}


/*#
    @class GdkEventButton
    @brief Used for button press and button release events.
    @param One type of the event-button types (GDK_BUTTON_PRESS, GDK_2BUTTON_PRESS, GDK_3BUTTON_PRESS or GDK_BUTTON_RELEASE).

    @prop type the type of the event (GDK_BUTTON_PRESS, GDK_2BUTTON_PRESS, GDK_3BUTTON_PRESS or GDK_BUTTON_RELEASE).
    @prop window TODO the window which received the event.
    @prop send_event TRUE if the event was sent explicitly (e.g. using XSendEvent).
    @prop time the time of the event in milliseconds.
    @prop x the x coordinate of the pointer relative to the window.
    @prop y the y coordinate of the pointer relative to the window.
    @prop axes TODO x, y translated to the axes of device, or NULL if device is the mouse.
    @prop state a bit-mask representing the state of the modifier keys (e.g. Control, Shift and Alt) and the pointer buttons. See GdkModifierType.
    @prop button the button which was pressed or released, numbered from 1 to 5. Normally button 1 is the left mouse button, 2 is the middle button, and 3 is the right button. On 2-button mice, the middle button can often be simulated by pressing both mouse buttons together.
    @prop device TODO the device where the event originated.
    @prop x_root the x coordinate of the pointer relative to the root of the screen.
    @prop y_root the y coordinate of the pointer relative to the root of the screen.

    @note In Falcon, this class inherits from GdkEvent.

    The type field will be one of GDK_BUTTON_PRESS, GDK_2BUTTON_PRESS, GDK_3BUTTON_PRESS,
    and GDK_BUTTON_RELEASE.

    Double and triple-clicks result in a sequence of events being received.
    For double-clicks the order of events will be:

        1. GDK_BUTTON_PRESS
        2. GDK_BUTTON_RELEASE
        3. GDK_BUTTON_PRESS
        4. GDK_2BUTTON_PRESS
        5. GDK_BUTTON_RELEASE

    Note that the first click is received just like a normal button press, while
    the second click results in a GDK_2BUTTON_PRESS being received just after the
    GDK_BUTTON_PRESS.

    Triple-clicks are very similar to double-clicks, except that GDK_3BUTTON_PRESS
    is inserted after the third click. The order of the events is:

        1. GDK_BUTTON_PRESS
        2. GDK_BUTTON_RELEASE
        3. GDK_BUTTON_PRESS
        4. GDK_2BUTTON_PRESS
        5. GDK_BUTTON_RELEASE
        6. GDK_BUTTON_PRESS
        7. GDK_3BUTTON_PRESS
        8. GDK_BUTTON_RELEASE

    For a double click to occur, the second button press must occur within 1/4 of a
    second of the first. For a triple click to occur, the third button press must
    also occur within 1/2 second of the first button press.
 */
FALCON_FUNC EventButton::init( VMARG )
{
    Item* i_tp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_tp || !i_tp->isInteger() )
        throw_inv_params( "GdkEventType" );
#endif
    // todo: check for correct event type
    MYSELF;
    self->setObject( (GdkEventButton*) gdk_event_new( (GdkEventType) i_tp->asInteger() ),
                    true );
}


} // Gdk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
