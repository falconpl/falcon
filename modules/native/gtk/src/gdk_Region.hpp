#ifndef GDK_REGION_HPP
#define GDK_REGION_HPP

#include "modgtk.hpp"

#define GET_REGION( item ) \
        (((Gdk::Region*) (item).asObjectSafe())->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Region
 */
class Region
    :
    public Gtk::VoidObject
{
public:

    Region( const Falcon::CoreClass*,
            const GdkRegion* = 0, const bool transfer = false );

    Region( const Region& );

    ~Region();

    Region* clone() const { return new Region( *this ); }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkRegion* getObject() const { return (GdkRegion*) m_obj; }

    //void setObject( const void* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC polygon( VMARG );

    static FALCON_FUNC copy( VMARG );

    static FALCON_FUNC rectangle( VMARG );

#if 0 // not used
    static FALCON_FUNC destroy( VMARG );
#endif

    static FALCON_FUNC get_clipbox( VMARG );

    static FALCON_FUNC get_rectangles( VMARG );

    static FALCON_FUNC empty( VMARG );

    static FALCON_FUNC equal( VMARG );

#if GTK_CHECK_VERSION( 2, 18, 0 )
    static FALCON_FUNC rect_equal( VMARG );
#endif

    static FALCON_FUNC point_in( VMARG );

    static FALCON_FUNC rect_in( VMARG );

    static FALCON_FUNC offset( VMARG );

    static FALCON_FUNC shrink( VMARG );

    static FALCON_FUNC union_with_rect( VMARG );

    static FALCON_FUNC intersect( VMARG );

    static FALCON_FUNC union_( VMARG );

    static FALCON_FUNC subtract( VMARG );

    static FALCON_FUNC xor_( VMARG );
#if 0 // todo
    static FALCON_FUNC spans_intersect_foreach( VMARG );
#endif

};


} // Gdk
} // Falcon

#endif // !GDK_REGION_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
