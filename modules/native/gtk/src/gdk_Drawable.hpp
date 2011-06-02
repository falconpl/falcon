#ifndef GDK_DRAWABLE_HPP
#define GDK_DRAWABLE_HPP

#include "modgtk.hpp"

#define GET_DRAWABLE( item ) \
        (((Gdk::Drawable*) (item).asObjectSafe() )->getObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Drawable
 */
class Drawable
    :
    public Gtk::CoreGObject
{
public:

    Drawable( const Falcon::CoreClass*, const GdkDrawable* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkDrawable* getObject() const { return (GdkDrawable*) m_obj; }

#if 0 // deprecated
    static FALCON_FUNC ref( VMARG );
    static FALCON_FUNC unref( VMARG );
    static FALCON_FUNC set_data( VMARG );
    static FALCON_FUNC get_data( VMARG );
#endif

    //static FALCON_FUNC get_display( VMARG );

    static FALCON_FUNC get_screen( VMARG );

    static FALCON_FUNC get_visual( VMARG );

    static FALCON_FUNC set_colormap( VMARG );

    static FALCON_FUNC get_colormap( VMARG );

    static FALCON_FUNC get_depth( VMARG );

    static FALCON_FUNC get_size( VMARG );

    static FALCON_FUNC get_clip_region( VMARG );

    static FALCON_FUNC get_visible_region( VMARG );

    static FALCON_FUNC draw_point( VMARG );

    static FALCON_FUNC draw_points( VMARG );

    static FALCON_FUNC draw_line( VMARG );
#if 0
    static FALCON_FUNC draw_lines( VMARG );

    static FALCON_FUNC draw_pixbuf( VMARG );

    static FALCON_FUNC draw_segments( VMARG );

    static FALCON_FUNC draw_rectangle( VMARG );

    static FALCON_FUNC draw_arc( VMARG );

    static FALCON_FUNC draw_polygon( VMARG );

    static FALCON_FUNC draw_trapezoids( VMARG );

    static FALCON_FUNC draw_glyphs( VMARG );

    static FALCON_FUNC draw_glyphs_transformed( VMARG );

    static FALCON_FUNC draw_layout_line( VMARG );

    static FALCON_FUNC draw_layout_line_with_colors( VMARG );

    static FALCON_FUNC draw_layout( VMARG );

    static FALCON_FUNC draw_layout_with_colors( VMARG );

    static FALCON_FUNC draw_string( VMARG );

    static FALCON_FUNC draw_text( VMARG );

    static FALCON_FUNC draw_text_wc( VMARG );

    static FALCON_FUNC draw_pixmap( VMARG );

    static FALCON_FUNC draw_drawable( VMARG );

    static FALCON_FUNC draw_image( VMARG );

    static FALCON_FUNC get_image( VMARG );

    static FALCON_FUNC copy_to_image( VMARG );
#endif

};


} // Gdk
} // Falcon

#endif // !GDK_DRAWABLE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
