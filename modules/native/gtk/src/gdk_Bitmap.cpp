/**
 *  \file gdk_Bitmap.cpp
 */

#include "gdk_Bitmap.hpp"

#include "gdk_Drawable.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Bitmap::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Bitmap = mod->addClass( "GdkBitmap", &Bitmap::create_from_data );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GdkDrawable" ) );
    c_Bitmap->getClassDef()->addInheritance( in );

    //c_Bitmap->setWKS( true );
    c_Bitmap->getClassDef()->factory( &Bitmap::factory );
}


Bitmap::Bitmap( const Falcon::CoreClass* gen, const GdkBitmap* bmap )
    :
    Gtk::CoreGObject( gen, (GObject*) bmap )
{}


Falcon::CoreObject* Bitmap::factory( const Falcon::CoreClass* gen, void* bmap, bool )
{
    return new Bitmap( gen, (GdkBitmap*) bmap );
}


/*#
    @class GdkBitmap
    @brief An opaque structure representing an offscreen drawable of depth 1.
    @param drawable a GdkDrawable, used to determine default values for the new pixmap. Can be NULL, in which case the root window is used.
    @param data a pointer to the XBM data.
    @param width the width of the new pixmap in pixels.
    @param height the height of the new pixmap in pixels.

    Pointers to structures of type GdkPixmap, GdkBitmap, and GdkWindow, can
    often be used interchangeably. The type GdkDrawable refers generically to
    any of these types.
 */
FALCON_FUNC Bitmap::create_from_data( VMARG )
{
    Item* i_draw = vm->param( 0 );
    Item* i_data = vm->param( 1 );
    Item* i_width = vm->param( 2 );
    Item* i_height = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_draw || !( i_draw->isNil() || ( i_draw->isObject()
        && IS_DERIVED( i_draw, GdkDrawable ) ) )
        || !i_data || !i_data->isString()
        || !i_width || !i_width->isInteger()
        || !i_height || !i_height->isInteger() )
        throw_inv_params( "[GdkDrawable],S,I,I" );
#endif
    AutoCString data( i_data->asString() );
    MYSELF;
    self->setObject( gdk_bitmap_create_from_data(
                        i_draw->isNil() ? NULL : GET_DRAWABLE( *i_draw ),
                        data.c_str(),
                        i_width->asInteger(),
                        i_height->asInteger() ) );
}


} // Gdk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
