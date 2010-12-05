/**
 *  \file gdk_Visual.cpp
 */

#include "gdk_Visual.hpp"

#include "gdk_Screen.hpp"

#undef MYSELF
#define MYSELF Gdk::Visual* self = Falcon::dyncast<Gdk::Visual*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Visual::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Visual = mod->addClass( "GdkVisual" );

    c_Visual->setWKS( true );
    c_Visual->getClassDef()->factory( &Visual::factory );

    //mod->addClassProperty( c_Visual, "parent_instance" );
    mod->addClassProperty( c_Visual, "type" );
    mod->addClassProperty( c_Visual, "depth" );
    mod->addClassProperty( c_Visual, "byte_order" );
    mod->addClassProperty( c_Visual, "colormap_size" );
    mod->addClassProperty( c_Visual, "bits_per_rgb" );
    mod->addClassProperty( c_Visual, "red_mask" );
    mod->addClassProperty( c_Visual, "red_shift" );
    mod->addClassProperty( c_Visual, "red_prec" );
    mod->addClassProperty( c_Visual, "green_mask" );
    mod->addClassProperty( c_Visual, "green_shift" );
    mod->addClassProperty( c_Visual, "green_prec" );
    mod->addClassProperty( c_Visual, "blue_mask" );
    mod->addClassProperty( c_Visual, "blue_shift" );
    mod->addClassProperty( c_Visual, "blue_prec" );

    Gtk::MethodTab methods[] =
    {
    { "query_depths",       &Visual::query_depths },
    { "query_visual_types", &Visual::query_visual_types },
    { "list_visuals",       &Visual::list_visuals },
    { "get_best_depth",     &Visual::get_best_depth },
    { "get_best_type",      &Visual::get_best_type },
    { "get_system",         &Visual::get_system },
    { "get_best",           &Visual::get_best },
    { "get_best_with_depth",&Visual::get_best_with_depth },
    { "get_best_with_type", &Visual::get_best_with_type },
    { "get_best_with_both", &Visual::get_best_with_both },
#if 0
    { "ref",                &Visual::ref },
    { "unref",              &Visual::unref },
#endif
    { "get_screen",         &Visual::get_screen },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Visual, meth->name, meth->cb );
}


Visual::Visual( const Falcon::CoreClass* gen, const GdkVisual* vis )
    :
    Gtk::VoidObject( gen, vis )
{
    if ( vis )
        incref();
}


Visual::Visual( const Visual& vis )
    :
    Gtk::VoidObject( vis )
{
    if ( vis.m_obj )
        incref();
}


Visual::~Visual()
{
    if ( m_obj )
        decref();
}


void Visual::incref() const
{
    assert( m_obj );
    gdk_visual_ref( (GdkVisual*) m_obj );
}


void Visual::decref() const
{
    assert( m_obj );
    gdk_visual_unref( (GdkVisual*) m_obj );
}


bool Visual::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkVisual* m_visual = (GdkVisual*) m_obj;

    if ( s == "type" )
        it = (int64) m_visual->type;
    else
    if ( s == "depth" )
        it = m_visual->depth;
    else
    if ( s == "byte_order" )
        it = (int64) m_visual->byte_order;
    else
    if ( s == "colormap_size" )
        it = m_visual->colormap_size;
    else
    if ( s == "bits_per_rgb" )
        it = m_visual->bits_per_rgb;
    else
    if ( s == "red_mask" )
        it = m_visual->red_mask;
    else
    if ( s == "red_shift" )
        it = m_visual->red_shift;
    else
    if ( s == "red_prec" )
        it = m_visual->red_prec;
    else
    if ( s == "green_mask" )
        it = m_visual->green_mask;
    else
    if ( s == "green_shift" )
        it = m_visual->green_shift;
    else
    if ( s == "green_prec" )
        it = m_visual->green_prec;
    else
    if ( s == "blue_mask" )
        it = m_visual->blue_mask;
    else
    if ( s == "blue_shift" )
        it = m_visual->blue_shift;
    else
    if ( s == "blue_prec" )
        it = m_visual->blue_prec;
    else
        return defaultProperty( s, it );
    return true;
}


bool Visual::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    assert( m_obj );
    GdkVisual* m_visual = (GdkVisual*) m_obj;

    if ( s == "type" )
        m_visual->type = (GdkVisualType) it.forceInteger();
    else
    if ( s == "depth" )
        m_visual->depth = it.forceInteger();
    else
    if ( s == "byte_order" )
        m_visual->byte_order = (GdkByteOrder) it.forceInteger();
    else
    if ( s == "colormap_size" )
        m_visual->colormap_size = it.forceInteger();
    else
    if ( s == "bits_per_rgb" )
        m_visual->bits_per_rgb = it.forceInteger();
    else
    if ( s == "red_mask" )
        m_visual->red_mask = it.forceInteger();
    else
    if ( s == "red_shift" )
        m_visual->red_shift = it.forceInteger();
    else
    if ( s == "red_prec" )
        m_visual->red_prec = it.forceInteger();
    else
    if ( s == "green_mask" )
        m_visual->green_mask = it.forceInteger();
    else
    if ( s == "green_shift" )
        m_visual->green_shift = it.forceInteger();
    else
    if ( s == "green_prec" )
        m_visual->green_prec = it.forceInteger();
    else
    if ( s == "blue_mask" )
        m_visual->blue_mask = it.forceInteger();
    else
    if ( s == "blue_shift" )
        m_visual->blue_shift = it.forceInteger();
    else
    if ( s == "blue_prec" )
        m_visual->blue_prec = it.forceInteger();
    else
        return false;
    return true;
}


Falcon::CoreObject* Visual::factory( const Falcon::CoreClass* gen, void* vis, bool )
{
    return new Visual( gen, (GdkVisual*) vis );
}


/*#
    @class GdkVisual
    @brief Low-level display hardware information

    A GdkVisual describes a particular video hardware display format.
    It includes information about the number of bits used for each color,
    the way the bits are translated into an RGB value for display,
    and the way the bits are stored in memory. For example, a piece of
    display hardware might support 24-bit color, 16-bit color, or 8-bit color;
    meaning 24/16/8-bit pixel sizes. For a given pixel size, pixels can be
    in different formats; for example the "red" element of an RGB pixel
    may be in the top 8 bits of the pixel, or may be in the lower 4 bits.

    Usually you can avoid thinking about visuals in GTK+. Visuals are useful
    to interpret the contents of a GdkImage, but you should avoid GdkImage
    precisely because its contents depend on the display hardware; use
    GdkPixbuf instead, for all but the most low-level purposes. Also, anytime
    you provide a GdkColormap, the visual is implied as part of the colormap
    (gdk_colormap_get_visual()), so you won't have to provide a visual in addition.

    There are several standard visuals. The visual returned by gdk_visual_get_system()
    is the system's default visual. gdk_rgb_get_visual() return the visual most
    suited to displaying full-color image data. If you use the calls in GdkRGB,
    you should create your windows using this visual (and the colormap returned by gdk_rgb_get_colormap()).

    A number of functions are provided for determining the "best" available visual.
    For the purposes of making this determination, higher bit depths are considered
    better, and for visuals of the same bit depth, GDK_VISUAL_PSEUDO_COLOR is
    preferred at 8bpp, otherwise, the visual types are ranked in the order of
    (highest to lowest) GDK_VISUAL_DIRECT_COLOR, GDK_VISUAL_TRUE_COLOR, GDK_VISUAL_PSEUDO_COLOR,
    GDK_VISUAL_STATIC_COLOR, GDK_VISUAL_GRAYSCALE, then GDK_VISUAL_STATIC_GRAY

    [...]

 */

/*#
    @method query_depths GdkVisual
    @brief This function returns the available bit depths for the default screen.
    @return An array of available depths.

    It's equivalent to listing the visuals (gdk_list_visuals()) and then looking
    at the depth field in each visual, removing duplicates.
 */
FALCON_FUNC Visual::query_depths( VMARG )
{
    NO_ARGS
    gint* depths;
    gint cnt;
    gdk_query_depths( &depths, &cnt );
    CoreArray* arr = new CoreArray( cnt );
    for ( int i = 0; i < cnt; ++i )
        arr->append( depths[i] );
    vm->retval( arr );
}


/*#
    @method query_visual_types GdkVisual
    @brief This function returns the available visual types for the default screen.
    @return An array of available visual types.

    It's equivalent to listing the visuals (gdk_list_visuals()) and then looking
    at the type field in each visual, removing duplicates.
 */
FALCON_FUNC Visual::query_visual_types( VMARG )
{
    NO_ARGS
    GdkVisualType* types;
    gint cnt;
    gdk_query_visual_types( &types, &cnt );
    CoreArray* arr = new CoreArray( cnt );
    for ( int i = 0; i < cnt; ++i )
        arr->append( types[i] );
    vm->retval( arr );
}


/*#
    @method list_visuals GdkVisual
    @brief Lists the available visuals for the default screen.
    @return An array of available visuals.

    (See gdk_screen_list_visuals()) A visual describes a hardware image data
    format. For example, a visual might support 24-bit color, or 8-bit color,
    and might expect pixels to be in a certain format.
 */
FALCON_FUNC Visual::list_visuals( VMARG )
{
    NO_ARGS
    GList* lst = gdk_list_visuals();
    GList* el;
    int cnt = 0;
    for ( el = lst; el; el = el->next ) ++cnt;
    CoreArray* arr = new CoreArray( cnt );
    for ( el = lst; el; el = el->next )
        arr->append( new Gdk::Visual(
                vm->findWKI( "GdkVisual" )->asClass(), (GdkVisual*) el->data ) );
    g_list_free( lst );
    vm->retval( arr );
}


/*#
    @method get_best_depth GdkVisual
    @brief Get the best available depth for the default GDK screen.
    @return best available depth

    "Best" means "largest," i.e. 32 preferred over 24 preferred over 8 bits per pixel.
 */
FALCON_FUNC Visual::get_best_depth( VMARG )
{
    NO_ARGS
    vm->retval( gdk_visual_get_best_depth() );
}


/*#
    @method get_best_type GdkVisual
    @brief Return the best available visual type for the default GDK screen.
    @return best visual type
 */
FALCON_FUNC Visual::get_best_type( VMARG )
{
    NO_ARGS
    vm->retval( (int64) gdk_visual_get_best_type() );
}


/*#
    @method get_system GdkVisual
    @brief Get the system's default visual for the default GDK screen.
    @return System GdkVisual.

    This is the visual for the root window of the display.
 */
FALCON_FUNC Visual::get_system( VMARG )
{
    NO_ARGS
    GdkVisual* vis = gdk_visual_get_system();
    vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(), vis ) );
}


/*#
    @method get_best GdkVisual
    @brief Get the visual with the most available colors for the default GDK screen.
    @return Best visual.
 */
FALCON_FUNC Visual::get_best( VMARG )
{
    NO_ARGS
    GdkVisual* vis = gdk_visual_get_best();
    vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(), vis ) );
}


/*#
    @method get_best_with_depth GdkVisual
    @brief Get the best visual with depth depth for the default GDK screen.
    @param depth a bit depth
    @return best visual for the given depth.

    Color visuals and visuals with mutable colormaps are preferred over grayscale
    or fixed-colormap visuals. Nil may be returned if no visual supports depth.
 */
FALCON_FUNC Visual::get_best_with_depth( VMARG )
{
    Item* i_depth = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_depth || i_depth->isNil() || !i_depth->isInteger() )
        throw_inv_params( "I" );
#endif
    GdkVisual* vis = gdk_visual_get_best_with_depth( i_depth->asInteger() );
    if ( vis )
        vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(), vis ) );
    else
        vm->retnil();
}


/*#
    @method get_best_with_type GdkVisual
    @brief Get the best visual of the given visual_type for the default GDK screen.
    @param visual_type a visual type
    @return best visual of the given type

    Visuals with higher color depths are considered better.
    Nil may be returned if no visual has type visual_type.
 */
FALCON_FUNC Visual::get_best_with_type( VMARG )
{
    Item* i_type = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_type || i_type->isNil() || !i_type->isInteger() )
        throw_inv_params( "I" );
#endif
    GdkVisual* vis = gdk_visual_get_best_with_type( (GdkVisualType) i_type->asInteger() );
    if ( vis )
        vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(), vis ) );
    else
        vm->retnil();
}


/*#
    @method get_best_with_both GdkVisual
    @brief Combines gdk_visual_get_best_with_depth() and gdk_visual_get_best_with_type().
    @param depth a bit depth
    @param visual_type a visual type
    @return best visual.
 */
FALCON_FUNC Visual::get_best_with_both( VMARG )
{
    Item* i_depth = vm->param( 0 );
    Item* i_type = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_depth || i_depth->isNil() || !i_depth->isInteger()
        || !i_type || i_type->isNil() || !i_type->isInteger() )
        throw_inv_params( "I,GdkVisualType" );
#endif
    GdkVisual* vis = gdk_visual_get_best_with_both(
            i_depth->asInteger(), (GdkVisualType) i_type->asInteger() );
    if ( vis )
        vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(), vis ) );
    else
        vm->retnil();
}


#if 0
FALCON_FUNC Visual::ref( VMARG );
FALCON_FUNC Visual::unref( VMARG );
#endif


/*#
    @method get_screen GdkVisual
    @brief Gets the screen to which this visual belongs
    @return the screen to which this visual belongs (GdkScreen)
 */
FALCON_FUNC Visual::get_screen( VMARG )
{
    NO_ARGS
    GdkScreen* scrn = gdk_visual_get_screen( GET_VISUAL( vm->self() ) );
    vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(), scrn ) );
}


} // Gdk
} // Falcon
