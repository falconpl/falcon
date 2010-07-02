#ifndef GDK_EVENT_HPP
#define GDK_EVENT_HPP

#include "modgtk.hpp"

#define GET_EVENT( item ) \
        (((Gdk::Event*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Event
 */
class Event
    :
    public Gtk::VoidObject
{
public:

    Event( const Falcon::CoreClass*, const GdkEvent* = 0, const bool transfer = false );

    Event( const Event& );

    virtual ~Event();

    virtual Event* clone() const { return new Event( *this ); }

    virtual bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    virtual bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkEvent* getObject() const { return (GdkEvent*) m_obj; }

    virtual void setObject( const void*, const bool transfer = false );

    static FALCON_FUNC get_real_event( VMARG );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC events_pending( VMARG );

    static FALCON_FUNC peek( VMARG );

    static FALCON_FUNC get( VMARG );

#if 0 // deprecated
    static FALCON_FUNC get_graphics_expose( VMARG );
#endif

    static FALCON_FUNC put( VMARG );

    static FALCON_FUNC copy( VMARG );

    static FALCON_FUNC get_time( VMARG );

    static FALCON_FUNC get_axis( VMARG );

    static FALCON_FUNC get_state( VMARG );

    static FALCON_FUNC get_coords( VMARG );

    static FALCON_FUNC get_root_coords( VMARG );

    static FALCON_FUNC get_show_events( VMARG );

    static FALCON_FUNC set_show_events( VMARG );

#if 0 // todo
    static FALCON_FUNC set_screen( VMARG );

    static FALCON_FUNC get_screen( VMARG );
#endif

};


} // Gdk
} // Falcon

#endif // !GDK_EVENT_HPP
