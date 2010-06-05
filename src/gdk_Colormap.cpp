/**
 *  \file gdk_Colormap.cpp
 */

#include "gdk_Colormap.hpp"

#include "gdk_Color.hpp"
#include "gdk_Screen.hpp"
#include "gdk_Visual.hpp"

#undef MYSELF
#define MYSELF Gdk::Colormap* self = Falcon::dyncast<Gdk::Colormap*>( vm->self().asObjectSafe() )


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Colormap::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Colormap = mod->addClass( "GdkColormap" );

    c_Colormap->setWKS( true );
    c_Colormap->getClassDef()->factory( &Colormap::factory );

    mod->addClassProperty( c_Colormap, "size" );
    mod->addClassProperty( c_Colormap, "colors" );

    Gtk::MethodTab methods[] =
    {
#if 0
    { "ref",            &Colormap::ref },
    { "unref",          &Colormap::unref },
#endif
    { "get_system",     &Colormap::get_system },
    { "get_system_size",&Colormap::get_system_size },
    { "change",         &Colormap::change },
#if 0
    { "alloc_colors",   &Colormap::alloc_colors },
    { "alloc_color",    &Colormap::alloc_color },
    { "free_colors",    &Colormap::free_colors },
    { "query_color",    &Colormap::query_color },
#endif
    { "get_visual",     &Colormap::get_visual },
    { "get_screen",     &Colormap::get_screen },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Colormap, meth->name, meth->cb );
}


Colormap::Colormap( const Falcon::CoreClass* gen, const GdkColormap* clr )
    :
    Falcon::CoreObject( gen )
{
    m_colormap = NULL;

    if ( clr )
    {
        m_colormap = (GdkColormap*) clr;
        gdk_colormap_ref( m_colormap );
    }
}


Colormap::~Colormap()
{
    if ( m_colormap )
        gdk_colormap_unref( m_colormap );
}


bool Colormap::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    if ( s == "size" )
        it = m_colormap->size;
    else
    if ( s == "colors" )
    { // todo
        if ( m_colormap )
        {
            CoreArray* arr = new CoreArray( m_colormap->size );
            Item* wki = VMachine::getCurrent()->findWKI( "GdkColor" );
            for ( int i = 0; i < m_colormap->size; ++i )
                arr->append( new Gdk::Color( wki->asClass(), &m_colormap->colors[i] ) );
            it = arr;
        }
        else
            return false;
    }
    else
        return defaultProperty( s, it );
    return true;
}


bool Colormap::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* Colormap::factory( const Falcon::CoreClass* gen, void* clr, bool )
{
    return new Colormap( gen, (GdkColormap*) clr );
}


/*#
    @class GdkColormap
    @brief The GdkColormap structure is used to describe an allocated or unallocated color.
    @param visual a GdkVisual.
    @param allocate if true, the newly created colormap will be a private colormap, and all colors in it will be allocated for the applications use.
    @return the new GdkColormap.

    @prop size For pseudo-color colormaps, the number of colors in the colormap.
    @prop colors An array containing the current values in the colormap. This can be used to map from pixel values back to RGB values. This is only meaningful for pseudo-color colormaps.
 */
FALCON_FUNC Colormap::init( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GdkVisual,B" );
    CoreObject* o_vis = args.getObject( 0 );
    gboolean b = args.getBoolean( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_vis, GdkVisual ) )
        throw_inv_params( "GdkVisual,B" );
#endif
    GdkVisual* vis = Falcon::dyncast<Gdk::Visual*>( o_vis )->getObject();
    MYSELF;
    GdkColormap* cmap = gdk_colormap_new( vis, b );
    self->m_colormap = cmap;
    gdk_colormap_ref( self->m_colormap );
}


#if 0
FALCON_FUNC Colormap::ref( VMARG );

FALCON_FUNC Colormap::unref( VMARG );
#endif


/*#
    @method get_system
    @brief Gets the system's default colormap for the default screen.
    @return the default colormap.
 */
FALCON_FUNC Colormap::get_system( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GdkColormap* cmap = gdk_colormap_get_system();
    vm->retval( new Gdk::Colormap( vm->findWKI( "GdkColormap" )->asClass(), cmap ) );
}


/*#
    @method get_system_size
    @brief Returns the size of the system's default colormap.
    @return the size of the system's default colormap.
 */
FALCON_FUNC Colormap::get_system_size( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    vm->retval( gdk_colormap_get_system_size() );
}


/*#
    @method change
    @brief Changes the value of the first ncolors in a private colormap to match the values in the colors array in the colormap.
    @param ncolors the number of colors to change.

    This function is obsolete and should not be used. See gdk_color_change().
 */
FALCON_FUNC Colormap::change( VMARG )
{
    Item* i_ncol = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ncol || i_ncol->isNil() || !i_ncol->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    gdk_colormap_change( self->m_colormap, i_ncol->asInteger() );
}


#if 0
FALCON_FUNC Colormap::alloc_colors( VMARG );

FALCON_FUNC Colormap::alloc_color( VMARG );

FALCON_FUNC Colormap::free_colors( VMARG );

FALCON_FUNC Colormap::query_color( VMARG );
#endif


/*#
    @method get_visual
    @brief Returns the visual for which a given colormap was created.
    @return the visual of the colormap.
 */
FALCON_FUNC Colormap::get_visual( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GdkVisual* vis = gdk_colormap_get_visual( self->m_colormap );
    vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(), vis ) );
}


/*#
    @method get_screen
    @brief Gets the screen for which this colormap was created.
    @return the screen for which this colormap was created.
 */
FALCON_FUNC Colormap::get_screen( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GdkScreen* scrn = gdk_colormap_get_screen( self->m_colormap );
    vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(), scrn ) );
}


} // Gdk
} // Falcon
