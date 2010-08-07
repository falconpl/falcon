/**
 *  \file gdk_Pixmap.cpp
 */

#include "gdk_Pixmap.hpp"

#include "gdk_Color.hpp"
#include "gdk_Drawable.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Pixmap::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Pixmap = mod->addClass( "GdkPixmap", &Pixmap::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GdkDrawable" ) );
    c_Pixmap->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    { "create_from_data",           &Pixmap::create_from_data },
#if 0 // todo
    { "create_from_xpm",            &Pixmap::create_from_xpm },
    { "colormap_create_from_xpm",   &Pixmap::colormap_create_from_xpm },
    { "create_from_xpm_d",          &Pixmap::create_from_xpm_d },
    { "colormap_create_from_xpm_d", &Pixmap::colormap_create_from_xpm_d },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Pixmap, meth->name, meth->cb );
}


Pixmap::Pixmap( const Falcon::CoreClass* gen, const GdkPixmap* pix )
    :
    Gtk::CoreGObject( gen, (GObject*) pix )
{}


Falcon::CoreObject* Pixmap::factory( const Falcon::CoreClass* gen, void* pix, bool )
{
    return new Pixmap( gen, (GdkPixmap*) pix );
}


/*#
    @class GdkPixmap
    @brief Offscreen drawables.
    @param drawable A GdkDrawable, used to determine default values for the new pixmap. Can be NULL if depth is specified.
    @param width The width of the new pixmap in pixels.
    @param height The height of the new pixmap in pixels.
    @param depth The depth (number of bits per pixel) of the new pixmap. If -1, and drawable is not NULL, the depth of the new pixmap will be equal to that of drawable.

    Pixmaps are offscreen drawables. They can be drawn upon with the standard
    drawing primitives, then copied to another drawable (such as a GdkWindow)
    with gdk_draw_drawable(). The depth of a pixmap is the number of bits per
    pixels. Bitmaps are simply pixmaps with a depth of 1. (That is, they are
    monochrome bitmaps - each pixel can be either on or off).
 */
FALCON_FUNC Pixmap::init( VMARG )
{
    Item* i_draw = vm->param( 0 );
    Item* i_width = vm->param( 1 );
    Item* i_height = vm->param( 2 );
    Item* i_depth = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_draw || !( i_draw->isNil() || ( i_draw->isObject()
        && IS_DERIVED( i_draw, GdkDrawable ) ) )
        || !i_width || !i_width->isInteger()
        || !i_height || !i_height->isInteger()
        || !i_depth || !i_depth->isInteger() )
        throw_inv_params( "[Gdkdrawable],I,I,I" );
    if ( i_depth->asInteger() == -1 && i_draw->isNil() )
        throw_inv_params( "Gdkdrawable == nil and depth == -1" );
#endif
    MYSELF;
    self->setObject( (GObject*) gdk_pixmap_new( i_draw->isNil() ? NULL : GET_DRAWABLE( *i_draw ),
                                                 i_width->asInteger(),
                                                 i_height->asInteger(),
                                                 i_depth->asInteger() ) );
}


/*#
    @method create_from_data
    @brief Create a two-color pixmap from data in XBM data.
    @param drawable a GdkDrawable, used to determine default values for the new pixmap. Can be NULL, if the depth is given.
    @param data the character data (string).
    @param width the width of the new pixmap in pixels.
    @param height the height of the new pixmap in pixels.
    @param depth the depth (number of bits per pixel) of the new pixmap.
    @param fg the foreground color (GdkColor).
    @param bg the background color (GdkColor).
    @return the GdkPixmap
 */
FALCON_FUNC Pixmap::create_from_data( VMARG )
{
    Item* i_draw = vm->param( 0 );
    Item* i_data = vm->param( 1 );
    Item* i_width = vm->param( 2 );
    Item* i_height = vm->param( 3 );
    Item* i_depth = vm->param( 4 );
    Item* i_fg = vm->param( 5 );
    Item* i_bg = vm->param( 6 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_draw || !( i_draw->isNil() || ( i_draw->isObject()
        && IS_DERIVED( i_draw, GdkDrawable ) ) )
        || !i_data || !i_data->isString()
        || !i_width || !i_width->isInteger()
        || !i_height || !i_height->isInteger()
        || !i_depth || !i_depth->isInteger()
        || !i_fg || !i_fg->isObject() || !IS_DERIVED( i_fg, GdkColor )
        || !i_bg || !i_bg->isObject() || !IS_DERIVED( i_bg, GdkColor ) )
        throw_inv_params( "[GdkDrawable],S,I,I,GdkColor,GdkColor" );
#endif
    AutoCString data( i_data->asString() );
    vm->retval( new Gdk::Pixmap( vm->findWKI( "GdkPixmap" )->asClass(),
                gdk_pixmap_create_from_data( i_draw->isNil() ? NULL : GET_DRAWABLE( *i_draw ),
                                             data.c_str(),
                                             i_width->asInteger(),
                                             i_height->asInteger(),
                                             i_depth->asInteger(),
                                             GET_COLOR( *i_fg ),
                                             GET_COLOR( *i_bg ) ) ) );
}


#if 0
FALCON_FUNC Pixmap::create_from_xpm( VMARG );

FALCON_FUNC Pixmap::colormap_create_from_xpm( VMARG );

FALCON_FUNC Pixmap::create_from_xpm_d( VMARG );

FALCON_FUNC Pixmap::colormap_create_from_xpm_d( VMARG );
#endif


} // Gdk
} // Falcon
