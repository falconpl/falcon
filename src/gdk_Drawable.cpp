/**
 *  \file gdk_Drawable.cpp
 */

#include "gdk_Drawable.hpp"

#include "gdk_Colormap.hpp"
//#include "gdk_Display.hpp"
#include "gdk_Region.hpp"
#include "gdk_Screen.hpp"
#include "gdk_Visual.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Drawable::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Drawable = mod->addClass( "GdkDrawable", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Drawable->getClassDef()->addInheritance( in );

    //c_Drawable->setWKS( true );
    c_Drawable->getClassDef()->factory( &Drawable::factory );

    Gtk::MethodTab methods[] =
    {
#if 0 // deprecated
    { "ref",                &Drawable::ref },
    { "unref",              &Drawable::unref },
    { "set_data",           &Drawable::set_data },
    { "get_data",           &Drawable::get_data },
#endif
    //{ "get_display",        &Drawable::get_display },
    { "get_screen",         &Drawable::get_screen },
    { "get_visual",         &Drawable::get_visual },
    { "set_colormap",       &Drawable::set_colormap },
    { "get_colormap",       &Drawable::get_colormap },
    { "get_depth",          &Drawable::get_depth },
    { "get_size",           &Drawable::get_size },
    { "get_clip_region",    &Drawable::get_clip_region },
    { "get_visible_region", &Drawable::get_visible_region },
#if 0
    { "draw_point",    &Drawable:: },
    { "draw_points",    &Drawable:: },
    { "draw_line",    &Drawable:: },
    { "draw_lines",    &Drawable:: },
    { "draw_pixbuf",    &Drawable:: },
    { "draw_segments",    &Drawable:: },
    { "draw_rectangle",    &Drawable:: },
    { "draw_arc",    &Drawable:: },
    { "draw_polygon",    &Drawable:: },
    { "draw_trapezoids",    &Drawable:: },
    { "draw_glyphs",    &Drawable:: },
    { "draw_glyphs_transformed",    &Drawable:: },
    { "draw_layout_line",    &Drawable:: },
    { "draw_layout_line_with_colors",    &Drawable:: },
    { "draw_layout",    &Drawable:: },
    { "draw_layout_with_colors",    &Drawable:: },
    { "draw_string",    &Drawable:: },
    { "draw_text",    &Drawable:: },
    { "draw_text_wc",    &Drawable:: },
    { "draw_pixmap",    &Drawable:: },
    { "draw_drawable",    &Drawable:: },
    { "draw_image",    &Drawable:: },
    { "get_image",    &Drawable:: },
    { "copy_to_image",    &Drawable:: },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Drawable, meth->name, meth->cb );
}


Drawable::Drawable( const Falcon::CoreClass* gen, const GdkDrawable* drawable )
    :
    Gtk::CoreGObject( gen, (GObject*) drawable )
{}


Falcon::CoreObject* Drawable::factory( const Falcon::CoreClass* gen, void* drawable, bool )
{
    return new Drawable( gen, (GdkDrawable*) drawable );
}


/*#
    @class GdkDrawable
    @brief Functions for drawing points, lines, arcs, and text

    These functions provide support for drawing points, lines, arcs and text onto
    what are called 'drawables'. Drawables, as the name suggests, are things
    which support drawing onto them, and are either GdkWindow or GdkPixmap objects.

    Many of the drawing operations take a GdkGC argument, which represents a
    graphics context. This GdkGC contains a number of drawing attributes such
    as foreground color, background color and line width, and is used to
    reduce the number of arguments needed for each drawing operation.
    See the Graphics Contexts section for more information.

    Some of the drawing operations take Pango data structures like PangoContext,
    PangoLayout or PangoLayoutLine as arguments. If you're using GTK+, the
    ususal way to obtain these structures is via
    gtk_widget_create_pango_context() or gtk_widget_create_pango_layout().
 */


#if 0 // deprecated
FALCON_FUNC Drawable::ref( VMARG );
FALCON_FUNC Drawable::unref( VMARG );
FALCON_FUNC Drawable::set_data( VMARG );
FALCON_FUNC Drawable::get_data( VMARG );
#endif


//FALCON_FUNC Drawable::get_display( VMARG );


/*#
    @method get_screen
    @brief Gets the GdkScreen associated with a GdkDrawable.
    @return the GdkScreen associated with drawable
 */
FALCON_FUNC Drawable::get_screen( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkScreen* screen = gdk_drawable_get_screen( (GdkDrawable*)_obj );
    vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(), screen ) );
}


/*#
    @method get_visual
    @brief Gets the GdkVisual describing the pixel format of drawable.
    @return a GdkVisual
 */
FALCON_FUNC Drawable::get_visual( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkVisual* vis = gdk_drawable_get_visual( (GdkDrawable*)_obj );
    vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(), vis ) );
}


/*#
    @method set_colormap
    @brief Sets the colormap associated with drawable.
    @param colormap a GdkColormap

    Normally this will happen automatically when the drawable is created; you on
    ly need to use this function if the drawable-creating function did not have
    a way to determine the colormap, and you then use drawable operations that
    require a colormap. The colormap for all drawables and graphics contexts you intend to use together should match. i.e. when using a GdkGC to draw to a drawable, or copying one drawable to another, the colormaps should match.
 */
FALCON_FUNC Drawable::set_colormap( VMARG )
{
    Item* i_map = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_map || !i_map->isObject() || !IS_DERIVED( i_map, GdkColormap ) )
        throw_inv_params( "GdkColormap" );
#endif
    GdkColormap* map = (GdkColormap*) COREGOBJECT( i_map )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gdk_drawable_set_colormap( (GdkDrawable*)_obj, map );
}


/*#
    @method get_colormap
    @brief Gets the colormap for drawable, if one is set; returns NULL otherwise.
    @return the colormap, or NULL
 */
FALCON_FUNC Drawable::get_colormap( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkColormap* map = gdk_drawable_get_colormap( (GdkDrawable*)_obj );
    if ( map )
        vm->retval( new Gdk::Colormap( vm->findWKI( "GdkColormap" )->asClass(), map ) );
    else
        vm->retnil();
}


/*#
    @method get_depth
    @brief Obtains the bit depth of the drawable, that is, the number of bits that make up a pixel in the drawable's visual.
    @return number of bits per pixel

    Examples are 8 bits per pixel, 24 bits per pixel, etc.
 */
FALCON_FUNC Drawable::get_depth( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gdk_drawable_get_depth( (GdkDrawable*)_obj ) );
}


/*#
    @method get_size
    @brief Returns the size of drawable.
    @return an array ( drawable's width, drawable's height )

    On the X11 platform, if drawable is a GdkWindow, the returned size is the
    size reported in the most-recently-processed configure event, rather than
    the current size on the X server.
 */
FALCON_FUNC Drawable::get_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gint w, h;
    gdk_drawable_get_size( (GdkDrawable*)_obj, &w, &h );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method get_clip_region
    @brief Computes the region of a drawable that potentially can be written to by drawing primitives.
    @return a GdkRegion.

    This region will not take into account the clip region for the GC, and may
    also not take into account other factors such as if the window is obscured
    by other windows, but no area outside of this region will be affected by
    drawing primitives.
 */
FALCON_FUNC Drawable::get_clip_region( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkRegion* reg = gdk_drawable_get_clip_region( (GdkDrawable*)_obj );
    vm->retval( new Gdk::Region( vm->findWKI( "GdkRegion" )->asClass(), reg,
                                 true ) );
}


/*#
    @method get_visible_region
    @brief Computes the region of a drawable that is potentially visible.
    @return a GdkRegion.

    This does not necessarily take into account if the window is obscured by
    other windows, but no area outside of this region is visible.
 */
FALCON_FUNC Drawable::get_visible_region( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkRegion* reg = gdk_drawable_get_visible_region( (GdkDrawable*)_obj );
    vm->retval( new Gdk::Region( vm->findWKI( "GdkRegion" )->asClass(), reg,
                                 true ) );
}


#if 0
FALCON_FUNC Drawable::draw_point( VMARG );
FALCON_FUNC Drawable::draw_points( VMARG );
FALCON_FUNC Drawable::draw_line( VMARG );
FALCON_FUNC Drawable::draw_lines( VMARG );
FALCON_FUNC Drawable::draw_pixbuf( VMARG );
FALCON_FUNC Drawable::draw_segments( VMARG );
FALCON_FUNC Drawable::draw_rectangle( VMARG );
FALCON_FUNC Drawable::draw_arc( VMARG );
FALCON_FUNC Drawable::draw_polygon( VMARG );
FALCON_FUNC Drawable::draw_trapezoids( VMARG );
FALCON_FUNC Drawable::draw_glyphs( VMARG );
FALCON_FUNC Drawable::draw_glyphs_transformed( VMARG );
FALCON_FUNC Drawable::draw_layout_line( VMARG );
FALCON_FUNC Drawable::draw_layout_line_with_colors( VMARG );
FALCON_FUNC Drawable::draw_layout( VMARG );
FALCON_FUNC Drawable::draw_layout_with_colors( VMARG );
FALCON_FUNC Drawable::draw_string( VMARG );
FALCON_FUNC Drawable::draw_text( VMARG );
FALCON_FUNC Drawable::draw_text_wc( VMARG );
FALCON_FUNC Drawable::draw_pixmap( VMARG );
FALCON_FUNC Drawable::draw_drawable( VMARG );
FALCON_FUNC Drawable::draw_image( VMARG );
FALCON_FUNC Drawable::get_image( VMARG );
FALCON_FUNC Drawable::copy_to_image( VMARG );
#endif

} // Gdk
} // Falcon
