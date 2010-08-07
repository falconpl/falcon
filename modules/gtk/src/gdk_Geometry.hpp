#ifndef GDK_GEOMETRY_HPP
#define GDK_GEOMETRY_HPP

#include "modgtk.hpp"

#define GET_GEOMETRY( item ) \
        (((Gdk::Geometry*) (item).asObjectSafe())->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Geometry
 */
class Geometry
    :
    public Gtk::VoidObject
{
public:

    Geometry( const Falcon::CoreClass*, const GdkGeometry* = 0 );

    Geometry( const Geometry& );

    ~Geometry();

    Geometry* clone() const { return new Geometry( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkGeometry* getObject() const { return (GdkGeometry*) m_obj; }

    void setObject( const void* );

    static FALCON_FUNC init( VMARG );

private:

    void alloc();

};


} // Gdk
} // Falcon

#endif // !GDK_GEOMETRY_HPP
