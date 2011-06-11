#ifndef GDK_POINT_HPP
#define GDK_POINT_HPP

#include "modgtk.hpp"

#define GET_POINT( item ) \
        (((Gdk::Point*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Point
 */
class Point
    :
    public Gtk::VoidObject
{
public:

    Point( const Falcon::CoreClass*, const GdkPoint* = 0 );

    Point( const Point& );

    ~Point();

    Point* clone() const { return new Point( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkPoint* getObject() const { return (GdkPoint*) m_obj; }

    void setObject( const void* );

    static FALCON_FUNC init( VMARG );

private:

    void alloc();

};


} // Gdk
} // Falcon

#endif // !GDK_POINT_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
