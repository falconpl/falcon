#ifndef GDK_RECTANGLE_HPP
#define GDK_RECTANGLE_HPP

#include "modgtk.hpp"

#define GET_RECTANGLE( item ) \
        (((Gdk::Rectangle*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Rectangle
 */
class Rectangle
    :
    public Gtk::VoidObject
{
public:

    Rectangle( const Falcon::CoreClass*, const GdkRectangle* = 0 );

    Rectangle( const Rectangle& );

    ~Rectangle();

    Rectangle* clone() const { return new Rectangle( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkRectangle* getObject() const { return (GdkRectangle*) m_obj; }

    void setObject( const void* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC intersect( VMARG );

    static FALCON_FUNC union_( VMARG );

private:

    void alloc();

};


} // Gdk
} // Falcon

#endif // !GDK_RECTANGLE_HPP
