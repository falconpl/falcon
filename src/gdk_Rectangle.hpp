#ifndef GDK_RECTANGLE_HPP
#define GDK_RECTANGLE_HPP

#include "modgtk.hpp"

#define GET_RECTANGLE( item ) \
        (Falcon::dyncast<Gdk::Rectangle*>( (item).asObjectSafe() )->getRectangle())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Rectangle
 */
class Rectangle
    :
    public Falcon::CoreObject
{
public:

    Rectangle( const Falcon::CoreClass*, const GdkRectangle* = 0 );

    ~Rectangle();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkRectangle* getRectangle() const { return (GdkRectangle*) &m_rectangle; }

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC intersect( VMARG );

    static FALCON_FUNC union_( VMARG );

private:

    GdkRectangle    m_rectangle;

};


} // Gdk
} // Falcon

#endif // !GDK_RECTANGLE_HPP
