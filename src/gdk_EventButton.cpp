/**
 *  \file gdk_EventButton.cpp
 */

#include "gdk_EventButton.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void EventButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_EventButton = mod->addClass( "GdkEventButton", &Gtk::abstract_init );

    c_EventButton->setWKS( true );
    c_EventButton->getClassDef()->factory( &EventButton::factory );

    mod->addClassProperty( c_EventButton, "type" );
    //mod->addClassProperty( c_EventButton, "window" );
    mod->addClassProperty( c_EventButton, "send_event" );
    mod->addClassProperty( c_EventButton, "time" );
    mod->addClassProperty( c_EventButton, "x" );
    mod->addClassProperty( c_EventButton, "y" );
    //mod->addClassProperty( c_EventButton, "axis" );
    mod->addClassProperty( c_EventButton, "state" );
    mod->addClassProperty( c_EventButton, "button" );
    //mod->addClassProperty( c_EventButton, "device" );
    mod->addClassProperty( c_EventButton, "x_root" );
    mod->addClassProperty( c_EventButton, "y_root" );
}


EventButton::EventButton( const Falcon::CoreClass* gen, const GdkEventButton* ev )
    :
    Falcon::CoreObject( gen )
{
    GdkEventButton* m_ev = (GdkEventButton*) memAlloc( sizeof( GdkEventButton ) );

    if ( !ev )
        memset( m_ev, 0, sizeof( GdkEventButton ) );
    else
        memcpy( m_ev, ev, sizeof( GdkEventButton ) );

    setUserData( m_ev );
}


EventButton::~EventButton()
{
    GdkEventButton* ev = (GdkEventButton*) getUserData();
    if ( ev )
        memFree( ev );
}


bool EventButton::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    GdkEventButton* m_ev = (GdkEventButton*) getUserData();

    if ( s == "type" )
        it = m_ev->type;
    else
    if ( s == "send_event" )
        it = (bool) m_ev->send_event;
    else
    if ( s == "time" )
        it = m_ev->time;
    else
    if ( s == "x" )
        it = m_ev->x;
    else
    if ( s == "y" )
        it = m_ev->y;
    else
    if ( s == "state" )
        it = m_ev->state;
    else
    if ( s == "button" )
        it = m_ev->button;
    else
    if ( s == "x_root" )
        it = m_ev->x_root;
    else
    if ( s == "y_root" )
        it = m_ev->y_root;
    else
        return false;
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


} // Gdk
} // Falcon
