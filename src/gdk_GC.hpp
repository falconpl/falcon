#ifndef GDK_GC_HPP
#define GDK_GC_HPP

#include "modgtk.hpp"

#define GET_GC( item ) \
        (((Gdk::GC*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::GC
 */
class GC
    :
    public Gtk::CoreGObject
{
public:

    GC( const Falcon::CoreClass*, const GdkGC* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkGC* getObject() const { return (GdkGC*) m_obj; }

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_values( VMARG );

    static FALCON_FUNC get_screen( VMARG );

#if 0 // unused
    static FALCON_FUNC ref( VMARG );
    static FALCON_FUNC unref( VMARG );
    static FALCON_FUNC destroy( VMARG );
#endif

    static FALCON_FUNC set_values( VMARG );

    static FALCON_FUNC get_values( VMARG );

#if 0 // todo
    static FALCON_FUNC set_foreground( VMARG );
    static FALCON_FUNC set_background( VMARG );
    static FALCON_FUNC set_rgb_fg_color( VMARG );
    static FALCON_FUNC set_rgb_bg_color( VMARG );
    static FALCON_FUNC set_font( VMARG );
    static FALCON_FUNC set_function( VMARG );
    static FALCON_FUNC set_fill( VMARG );
    static FALCON_FUNC set_tile( VMARG );
    static FALCON_FUNC set_stipple( VMARG );
    static FALCON_FUNC set_ts_origin( VMARG );
    static FALCON_FUNC set_clip_origin( VMARG );
    static FALCON_FUNC set_clip_mask( VMARG );
    static FALCON_FUNC set_clip_rectangle( VMARG );
    static FALCON_FUNC set_clip_region( VMARG );
    static FALCON_FUNC set_subwindow( VMARG );
    static FALCON_FUNC set_exposures( VMARG );
    static FALCON_FUNC set_line_attributes( VMARG );
    static FALCON_FUNC set_dashes( VMARG );
    static FALCON_FUNC copy( VMARG );
    static FALCON_FUNC set_colormap( VMARG );
    static FALCON_FUNC get_colormap( VMARG );
    static FALCON_FUNC offset( VMARG );
#endif
};


} // Gdk
} // Falcon

#endif // !GDK_GC_HPP
