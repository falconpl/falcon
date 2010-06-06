#ifndef GDK_EVENTBUTTON_HPP
#define GDK_EVENTBUTTON_HPP

#include "modgtk.hpp"

#include "gdk_Event.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::EventButton
 */
class EventButton
    :
    public Gdk::Event
{
public:

    EventButton( const Falcon::CoreClass*,
                 const GdkEventButton* = 0, const bool transfer = false );

    ~EventButton();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkEventButton* getEvent() const { return (GdkEventButton*) m_event; }

    void setEvent( const GdkEventButton*, const bool transfer = false );

    static FALCON_FUNC init( VMARG );

};


} // Gdk
} // Falcon

#endif // !GDK_EVENTBUTTON_HPP
