/**
 *  \file gdk_Colormap.cpp
 */

#include "gdk_Colormap.hpp"

#include "gdk_Color.hpp"
#include "gdk_Screen.hpp"
#include "gdk_Visual.hpp"

#undef MYSELF
#define MYSELF Gdk::Colormap* self = Falcon::dyncast<Gdk::Colormap*>( vm->self().asObjectSafe() )

/*#
   @beginmodule gtk
*/

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
    Gtk::VoidObject( gen, clr )
{
    incref();
}


Colormap::Colormap( const Colormap& other )
    :
    Gtk::VoidObject( other )
{
    incref();
}


Colormap::~Colormap()
{
    decref();
}


void Colormap::incref() const
{
    if ( m_obj )
        gdk_colormap_ref( (GdkColormap*) m_obj );
}


void Colormap::decref() const
{
    if ( m_obj )
        gdk_colormap_unref( (GdkColormap*) m_obj );
}


void Colormap::setObject( const void* map )
{
    VoidObject::setObject( map );
    incref();
}


bool Colormap::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GdkColormap* m_colormap = (GdkColormap*) m_obj;

    if ( s == "size" )
        it = m_colormap->size;
    else
    if ( s == "colors" )
    {
        CoreArray* arr = new CoreArray( m_colormap->size );
        Item* wki = VMachine::getCurrent()->findWKI( "GdkColor" );
        for ( int i = 0; i < m_colormap->size; ++i )
            arr->append( new Gdk::Color( wki->asClass(), &m_colormap->colors[i] ) );
        it = arr;
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

    @prop size For pseudo-color colormaps, the number of colors in the colormap.
    @prop colors An array containing the current values in the colormap. This can be used to map from pixel values back to RGB values. This is only meaningful for pseudo-color colormaps.
 */
FALCON_FUNC Colormap::init( VMARG )
{
    Item* i_vis = vm->param( 0 );
    Item* i_allocate = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_vis || !i_vis->isObject() || !IS_DERIVED( i_vis, GdkVisual )
        || !i_allocate || !i_allocate->isBoolean() )
        throw_inv_params( "GdkVisual,B" );
#endif
    MYSELF;
    self->setObject( gdk_colormap_new( GET_VISUAL( *i_vis ),
                                       (gboolean) i_allocate->asBoolean() ) );
}


#if 0
FALCON_FUNC Colormap::ref( VMARG );

FALCON_FUNC Colormap::unref( VMARG );
#endif


/*#
    @method get_system GdkColormap
    @brief Gets the system's default colormap for the default screen.
    @return the default colormap.
 */
FALCON_FUNC Colormap::get_system( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Colormap( vm->findWKI( "GdkColormap" )->asClass(),
                                   gdk_colormap_get_system() ) );
}


/*#
    @method get_system_size GdkColormap
    @brief Returns the size of the system's default colormap.
    @return the size of the system's default colormap.
 */
FALCON_FUNC Colormap::get_system_size( VMARG )
{
    NO_ARGS
    vm->retval( gdk_colormap_get_system_size() );
}


/*#
    @method change GdkColormap
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
    gdk_colormap_change( GET_COLORMAP( vm->self() ), i_ncol->asInteger() );
}


#if 0
FALCON_FUNC Colormap::alloc_colors( VMARG );

FALCON_FUNC Colormap::alloc_color( VMARG );

FALCON_FUNC Colormap::free_colors( VMARG );

FALCON_FUNC Colormap::query_color( VMARG );
#endif


/*#
    @method get_visual GdkColormap
    @brief Returns the visual for which a given colormap was created.
    @return the visual of the colormap.
 */
FALCON_FUNC Colormap::get_visual( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Visual( vm->findWKI( "GdkVisual" )->asClass(),
                        gdk_colormap_get_visual( GET_COLORMAP( vm->self() ) ) ) );
}


/*#
    @method get_screen GdkColormap
    @brief Gets the screen for which this colormap was created.
    @return the screen for which this colormap was created.
 */
FALCON_FUNC Colormap::get_screen( VMARG )
{
    NO_ARGS
    vm->retval( new Gdk::Screen( vm->findWKI( "GdkScreen" )->asClass(),
                        gdk_colormap_get_screen( GET_COLORMAP( vm->self() ) ) ) );
}


} // Gdk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
