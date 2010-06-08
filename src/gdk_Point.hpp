#ifndef GDK_POINT_HPP
#define GDK_POINT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Point
 */
class Point
    :
    public Falcon::CoreObject
{
public:

    Point( const Falcon::CoreClass*, const GdkPoint* = 0 );

    ~Point();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkPoint* getPoint() const { return (GdkPoint*) &m_point; }

    static FALCON_FUNC init( VMARG );

private:

    GdkPoint    m_point;

};


} // Gdk
} // Falcon

#endif // !GDK_POINT_HPP
