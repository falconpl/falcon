#ifndef GDK_EVENTBUTTON_HPP
#define GDK_EVENTBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gtk::EventButton
 */
class EventButton
    :
    public Falcon::CoreObject
{
public:

    EventButton( const Falcon::CoreClass*, const GdkEventButton* = 0 );

    ~EventButton();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkEventButton* getObject() const { return (GdkEventButton*) &m_eventButton; }

private:

    GdkEventButton  m_eventButton;

};


} // Gdk
} // Falcon

#endif // !GDK_EVENTBUTTON_HPP
