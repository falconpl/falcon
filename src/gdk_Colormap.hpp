#ifndef GDK_COLORMAP_HPP
#define GDK_COLORMAP_HPP

#include "modgtk.hpp"

#define GET_COLORMAP( item ) \
        (Falcon::dyncast<Gdk::Colormap*>( (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Colormap
 */
class Colormap
    :
    public Falcon::CoreObject
{
public:

    Colormap( const Falcon::CoreClass*, const GdkColormap* = 0 );

    ~Colormap();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );
#if 0
    static FALCON_FUNC ref( VMARG );

    static FALCON_FUNC unref( VMARG );
#endif
    static FALCON_FUNC get_system( VMARG );

    static FALCON_FUNC get_system_size( VMARG );

    static FALCON_FUNC change( VMARG );

    static FALCON_FUNC alloc_colors( VMARG );
#if 0
    static FALCON_FUNC alloc_color( VMARG );

    static FALCON_FUNC free_colors( VMARG );

    static FALCON_FUNC query_color( VMARG );
#endif
    static FALCON_FUNC get_visual( VMARG );

    static FALCON_FUNC get_screen( VMARG );


    GdkColormap* getObject() const { return m_colormap; }

private:

    GdkColormap*   m_colormap;

};


} // Gdk
} // Falcon

#endif // !GDK_COLORMAP_HPP
