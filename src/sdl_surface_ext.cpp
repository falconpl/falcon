/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_surface_ext.cpp

   The SDL binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "sdl_ext.h"
#include "sdl_mod.h"

#include <SDL.h>

/*# @beginmodule fsdl */

namespace Falcon {
namespace Ext {


/*#
   @class SDLSurface
   @brief Encapsulates SDL_Surface structures and provides related services.

   This class is used to store SDL_Surface C structure that is widely used by SDL
   graphics engine. This class also provides the vast majority of the functions
   that operates on SDL surfaces as methods.

   Falcon provides a subclass called @a SDLScreen that provides also screen oriented
   functions; SDL do not differentiates between surfaces being used to handle image
   buffers and surfaces used as screen handles, but Falcon embedding provides a
   differentiation both for OOP cleanness and to minimize memory usage (i.e. to store method
   pointers unneeded by the vast majority of surfaces).

   @prop w Width of the surface.
   @prop h Height of the surface.
   @prop pitch Bytes per line; see remarks on @a SDLSurface.pixels.
   @prop clip_rect An object of class @a SDLRect holding the clip rectangle (read only)
   @prop bpp Shortcut to access format.BytesPerPixel.
   @prop format @a SDLPixelFormat instance describing the pixels in this surface.
*/

/*#
   @property pixels SDLSurface
   @brief MemBuf storage for pixels of this surface.

   This property allows to read and write from a surface memory buffer.
   Each entry is already pointing to an item of the correct size to represent
   pixels on this surface. So, if the surface is large 640 pixels, the correct
   size of a line is 640 elements, regardless of the number of bytes per pixels.
   It is possible to retreive the count of elements for a line dividing
   SDLSurface.pitch by SDLSurface.bpp.

   Remember to call @a SDLSurface.LockIfNeeded before using this property if
   the object may be used by another thread in your application (or if it is
   a shared object as a screen).
*/

/*#
   @property flags SDLSurface
   @brief SDL flags for this surface.

   They can be a combination of the following:
   - SDL.SWSURFACE
   - SDL.HWSURFACE
   - SDL.ASYNCBLIT
   - SDL.ANYFORMAT
   - SDL.HWPALETTE
   - SDL.DOUBLEBUF
   - SDL.FULLSCREEN
   - SDL.OPENGL
   - SDL.OPENGLBLIT
   - SDL.RESIZABLE
   - SDL.NOFRAME
   - SDL.HWACCEL
   - SDL.SRCCOLORKEY
   - SDL.RLEACCEL
   - SDL.SRCALPHA
   - SDL.PREALLOC
*/

/*#
   @method SaveBMP SDLSurface
   @brief Saves a BMP files to disk.
   @param filename the file where to store this BMP.
   @raise SDLError on failure.

   Save a memory image (or even a screenshot, if this surface is also a screen)
   to a disk BMP file.

   The filename is totally subject to SDL rules, as it is simply passed
   to SDL engine. No Falcon URI parsing is perfomred before invoking SDL;
   as such, it is advisable to use this function only in simple applications.

   @see SDL.LoadBMP
*/
FALCON_FUNC SDLSurface_SaveBMP( ::Falcon::VMachine *vm )
{
   Item *i_file = vm->param(0);
   if ( i_file == 0 || ! i_file->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   AutoCString fname( *i_file->asString() );

   CoreObject *self = vm->self().asObject();
   SDL_Surface *source = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();

   if ( ::SDL_SaveBMP( source, fname.c_str() ) < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 5, __LINE__ )
         .desc( "SDL SaveBMP" )
         .extra( SDL_GetError() ) ) );
      return;
   }
   vm->retnil();
}

/*#
   @method BlitSurface SDLSurface
   @brief Copies a surface of part of it onto another surface.
   @param srcRect a @a SDLRect containing the source coordinates or nil.
   @param dest the destionation SDLSurface.
   @optparam dstRect a @a SDLRect containing destination coordinates or nil.
   @raise SDLError on copy failure.

   This functions copies a part of an image into another. The srcRect parameter determines
   which portion of the source image is copied; if nil, the whole image will be used.
   Only x and y coordinates from dstRect are used; if not provided, the image is copied
   starting at 0,0 (upper left corner).

*/

FALCON_FUNC SDLSurface_BlitSurface( ::Falcon::VMachine *vm )
{
   // rects can be nil, in which case we must set them to null
   Item *i_srcrect = vm->param(0);
   Item *i_dest = vm->param(1);
   Item *i_dstrect = vm->param(2);

   bool paramOk = true;
   SDL_Rect srcRect, dstRect, *pSrcRect, *pDstRect;

   if ( ( i_srcrect == 0 || ( ! i_srcrect->isNil() && ! i_srcrect->isObject() )) ||
        ( i_dest == 0 || ! i_dest->isObject() || ! i_dest->asObject()->derivedFrom( "SDLSurface" )) ||
        ( i_dstrect != 0 && ! i_dstrect->isNil() && ! i_dstrect->isObject() )
      )
   {
      paramOk = false;
   }
   else
   {
      // are rects exposing the right interface?
      if( i_srcrect != 0 && i_srcrect->isObject() )
      {
         if ( ! ObjectToRect( i_srcrect->asObject(), srcRect ) )
            paramOk = false;
         else
            pSrcRect = &srcRect;
      }
      else
         pSrcRect = 0;

      if ( paramOk )
      {
         if( i_dstrect != 0 && i_dstrect->isObject() )
         {
            if ( ! ObjectToRect( i_dstrect->asObject(), dstRect ) )
               paramOk = false;
            else
               pDstRect = &dstRect;
         }
         else
            pDstRect = 0;
      }
   }

   // are our parameter ok?
   if ( ! paramOk )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "SDLRect|Nil, SDLSurface [, SDLRect|Nil]" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *source = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   SDL_Surface *dest = static_cast<SDLSurfaceCarrier_impl*>( i_dest->asObject()->getUserData() )->surface();

   int res = ::SDL_BlitSurface( source, pSrcRect, dest, pDstRect );
   if( res < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 4, __LINE__ )
         .desc( "SDL BlitSurface" )
         .extra( SDL_GetError() ) ) );
      return;
   }
}

/*#
   @method SetPixel SDLSurface
   @brief Sets a single pixel to desired value
   @param x X coordinates of the pixel to be set
   @param y Y coordinates of the pixel to be set
   @param value The value to be set
   @raise ParamError if x or y are out of ranges

   This functions sets the color of a pixel to the desired value.
   The value is the palette index if this map has a palette,
   otherwise is a truecolor value whose precision depends on the
   mode depth.

   To get a suitable value for this surface,
   use @a SDLSurface.GetPixel.
*/

FALCON_FUNC SDLSurface_SetPixel( ::Falcon::VMachine *vm )
{
   Item *i_x = vm->param(0);
   Item *i_y = vm->param(1);
   Item *i_value = vm->param(2);

   if ( i_x == 0 || ! i_x->isOrdinal() ||
        i_y == 0 || ! i_y->isOrdinal() ||
        i_value == 0 || ! i_value->isOrdinal()
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   int x = (int) i_x->forceInteger();
   int y = (int) i_y->forceInteger();

   if ( x < 0 || x >= surface->w || y < 0 || y >= surface->h )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ ) ) );
      return;
   }

   Uint32 pixel = (Uint32) i_value->forceInteger();

   int bpp = surface->format->BytesPerPixel;
   /* Here p is the address to the pixel we want to set */
   Uint8 *p = (Uint8 *) surface->pixels + y * surface->pitch + x * bpp;

    switch(bpp) {
    case 1:
        *p = pixel;
        break;

    case 2:
        *(Uint16 *)p = pixel;
        break;

    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
            p[0] = (pixel >> 16) & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = pixel & 0xff;
        } else {
            p[0] = pixel & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = (pixel >> 16) & 0xff;
        }
        break;

    case 4:
        *(Uint32 *)p = pixel;
        break;
    }
}

/*#
   @method GetPixel SDLSurface
   @brief Get a single pixel value from the surface
   @param x X coordinates of the pixel to be retreived
   @param y Y coordinates of the pixel to be retreived
   @raise ParamError if x or y are out of range

   This functions gets the color of a pixel.
   The value is the palette index if this map has a palette,
   otherwise is a truecolor value whose precision depends on the
   mode depth.

   To determine the RGBA values of this pixel, use SDLSurface.GetPixelComponents.
*/

FALCON_FUNC SDLSurface_GetPixel( ::Falcon::VMachine *vm )
{
   Item *i_x = vm->param(0);
   Item *i_y = vm->param(1);

   if ( i_x == 0 || ! i_x->isOrdinal() ||
        i_y == 0 || ! i_y->isOrdinal()
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   int x = (int) i_x->forceInteger();
   int y = (int) i_y->forceInteger();

   if ( x < 0 || x >= surface->w || y < 0 || y >= surface->h )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ ) ) );
      return;
   }

   int bpp = surface->format->BytesPerPixel;
   /* Here p is the address to the pixel we want to retrieve */
   Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

   switch(bpp) {
   case 1:
      vm->retval( (int64) *p );
      break;

   case 2:
      vm->retval( (int64) ( *(Uint16 *)p));
      break;

   case 3:
      if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
         vm->retval( (int64) ( p[0] << 16 | p[1] << 8 | p[2]));
      else
         vm->retval( (int64) ( p[0] | p[1] << 8 | p[2] << 16));
      break;

   case 4:
      vm->retval( (int64) (*(Uint32 *)p));
      break;

   default:
      vm->retval(0);       //
   }
}

/*#
   @method GetPixelIndex SDLSurface
   @brief Return the index of a pixel in the pixels array of this class
   @param x X coordinates of the desired pixel position
   @param y Y coordinates of the desired pixel position
   @raise ParamError if x or y are out of range

   This is just a shortcut for the formula
   \code
      index = x * bpp + y * pitch
   \endcode

   Through the direct index, it is possible to change or read directly
   a pixel in the pixels property of this class (which holds a correctly
   sized MemBuf).
*/

FALCON_FUNC SDLSurface_GetPixelIndex( ::Falcon::VMachine *vm )
{
   Item *i_x = vm->param(0);
   Item *i_y = vm->param(1);

   if ( i_x == 0 || ! i_x->isOrdinal() ||
        i_y == 0 || ! i_y->isOrdinal()
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   int x = (int) i_x->forceInteger();
   int y = (int) i_y->forceInteger();

   if ( x < 0 || x >= surface->w || y < 0 || y >= surface->h )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ ) ) );
      return;
   }

   int bpp = surface->format->BytesPerPixel;
   vm->retval( (int64) y * surface->pitch/bpp + x );
}

/*#
   @method LockSurface SDLSurface
   @brief Locks the surface for low level byte access.

   This allows to mangle with internal surface and format bits. An internal counter of
   locks is kept, and the surface is unlocked correctly at destruction, if needed.

   So, it is still possible to interrupt VM and destroy it while holding locks on
   surfaces. However, please notice that SDL Locks are quite invasive, use them
   sparcely and only for the needed operations.

   Possibly, use the @a SDLSurface.LockIfNeeded Falcon specific extension.
*/
FALCON_FUNC SDLSurface_LockSurface( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDLSurfaceCarrier_impl *carrier = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() );
   SDL_LockSurface( carrier->surface() );
   carrier->m_lockCount++;
}

/*#
   @method UnlockSurface SDLSurface
   @brief Unlocks the surface.

   Possibly, use the @a SDLSurface.UnlockIfNeeded Falcon specific extension.
*/
FALCON_FUNC SDLSurface_UnlockSurface( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDLSurfaceCarrier_impl *carrier = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() );
   carrier->m_lockCount--;
   SDL_UnlockSurface( carrier->surface() );
}

/*#
   @method LockIfNeeded SDLSurface
   @brief Locks a surface for deep binary access only if needed.
*/
FALCON_FUNC SDLSurface_LockIfNeeded( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDLSurfaceCarrier_impl *carrier = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() );
   if( SDL_MUSTLOCK( carrier->surface() ) )
   {
      SDL_LockSurface( carrier->surface() );
      carrier->m_lockCount++;
   }
}

/*#
   @method UnlockIfNeeded SDLSurface
   @brief Unlocks a surface for deep binary access only if needed.
*/
FALCON_FUNC SDLSurface_UnlockIfNeeded( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDLSurfaceCarrier_impl *carrier = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() );
   if( SDL_MUSTLOCK( carrier->surface() ) )
   {
      carrier->m_lockCount--;
      SDL_UnlockSurface( carrier->surface() );
   }
}

/*#
   @method IsLockNeeded SDLSurface
   @brief Tells wether locks are needed on this surface or not.
   @return true if the user should lock this surface before accessing its binary
   buffers.
*/
FALCON_FUNC SDLSurface_IsLockNeeded( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   vm->retval( SDL_MUSTLOCK( surface ) ? true : false );
}

/*#
   @method FillRect SDLSurface
   @brief Fills a rectangle with a given color.
   @param rect an SDLRect instance containing the coordinates to fill, or nil to fill all
   @param color a color value to be used in fills.
   @raise SDLError on error
*/
FALCON_FUNC SDLSurface_FillRect( ::Falcon::VMachine *vm )
{
   Item *i_rect = vm->param(0);
   Item *i_color = vm->param(1);

   SDL_Rect rect;

   if ( i_rect == 0 || i_color == 0 || ! i_color->isOrdinal() ||
      ( ! i_rect->isNil() && ! ( i_rect->isObject() && ObjectToRect( i_rect->asObject(), rect ) ) )
   )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "SDLRect|Nil, N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   SDL_Rect *pRect = i_rect->isNil() ? 0 : &rect;

   if ( ::SDL_FillRect( surface, pRect, (Uint32) i_color->forceInteger() ) != 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 6, __LINE__ )
         .desc( "SDL FillRect error" )
         .extra( SDL_GetError() ) ) );
   }
}

/*#
   @method GetRGBA SDLSurface
   @brief Decomposes a given pixel value to RGBA values.
   @param color multibyte value of a color
   @optparam retArray An array that is used to store the desired values.
   @return a 4 element array (Red, Green, Blue and Alpha).
   @raise ParamError if color is out of index in palette based images

   This method is meant to determine the value of each component in a
   palette or truecolor value that has been read on a surface with
   compatible pixel format.
*/
FALCON_FUNC SDLSurface_GetRGBA( ::Falcon::VMachine *vm )
{
   Item *i_color = vm->param(0);
   Item *i_array = vm->param(1);

   if ( i_color == 0 || ! i_color->isOrdinal() ||
        ( i_array != 0 && ! i_array->isArray() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   SDL_PixelFormat *fmt = surface->format;

   Uint32 color = (Uint32) i_color->forceInteger();

   CoreArray *array = i_array == 0 ? new CoreArray( vm, 4 ) : i_array->asArray();
   array->length(0);
   Uint8 r, g, b, a;
   SDL_GetRGBA( color, fmt, &r, &b, &g, &a );
   array->append( (int64) r );
   array->append( (int64) g );
   array->append( (int64) b );
   array->append( (int64) a );

   vm->retval( array );
}

/*#
   @method SetAlpha SDLSurface
   @brief Sets the ALPHA settings for this surface
   @param flags ALPHA composition flags.
   @param alpha A value between 0 (transparent) and 255 (opaque)
   @raise SDLError on error.

   SDL_SetAlpha is used for setting the per-surface alpha value and/or enabling and disabling alpha blending.

   The @b flags is used to specify whether alpha blending should be used (SDL.SRCALPHA)
   and whether the surface should use RLE acceleration for blitting (SDL.RLEACCEL).
   The @b flags can be an OR'd combination of these two options, one of these options or 0.
   If SDL.SRCALPHA is not passed as a flag then all alpha information is ignored when blitting the surface.

   The @b alpha parameter is the per-surface alpha value; a surface need not
   have an alpha channel to use per-surface alpha and blitting can still
   be accelerated with SDL_RLEACCEL.
*/
FALCON_FUNC SDLSurface_SetAlpha( ::Falcon::VMachine *vm )
{
   Item *i_flags = vm->param(0);
   Item *i_alpha = vm->param(1);

   if ( i_flags == 0 || ! i_flags->isOrdinal() ||
        i_alpha == 0 || ! i_alpha->isOrdinal()
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "I,I" ) ) );
      return;
   }

   uint32 flags = (uint32) i_flags->forceInteger();
   byte alpha = (int8) i_alpha->forceInteger();

   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   
   if ( ::SDL_SetAlpha( surface, flags, alpha ) != 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 10, __LINE__ )
         .desc( "SDL SetAlpha error" )
         .extra( SDL_GetError() ) ) );
   }
}

/*#
   @method MapRGBA SDLSurface
   @brief Builds a color value from RGBA values.
   @param red Red component of the final color
   @param green Green component of the final color
   @param blue Blue component of the final color
   @optparam alpha Alpha component of the final color
   @return an integer that can be directly stored as color in this image.

   It is just a shortcut to properly perfomed shifts and binary operations.
*/
FALCON_FUNC SDLSurface_MapRGBA( ::Falcon::VMachine *vm )
{
   Item *i_red = vm->param(0);
   Item *i_green = vm->param(1);
   Item *i_blue = vm->param(2);
   Item *i_alpha = vm->param(3);

   if ( i_red == 0 || ! i_red->isOrdinal() ||
        i_green == 0 || ! i_green->isOrdinal() ||
        i_blue == 0 || ! i_blue->isOrdinal() ||
        (i_alpha != 0 && ! i_alpha->isOrdinal())
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,N,N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   SDL_PixelFormat *fmt = surface->format;

   Uint8 r = (Uint8) i_red->forceInteger();
   Uint8 g = (Uint8) i_green->forceInteger();
   Uint8 b = (Uint8) i_blue->forceInteger();
   if( i_alpha == 0 )
   {
      vm->retval( (int64) SDL_MapRGB( fmt, r, g, b ) );
   }
   else
   {
      vm->retval( (int64) SDL_MapRGBA( fmt, r, g, b,  (Uint8) i_alpha->forceInteger() ) );
   }
}


/*#
   @method SetColors SDLSurface
   @brief Changes part of the color map of a palettized surface.
   @param colors A 4 bytes per entry MemBuf containing the color data.
   @param firstColor An integer specifying the first color of the palette.
   @return true if the palette could be changed, 0 if the image has no palette or
           if the color map was too wide to fit the image palette.

   The colors in the MemBuf are stored with red, green, blue stored respectively at
   smallest, medium and highest address (regardless of endianity). In other words,
   the first 8 bits of the color number are the red value, the next 8 bits are the
   green value and the next 8 bits are blue.
*/
FALCON_FUNC SDLSurface_SetColors( ::Falcon::VMachine *vm )
{
   Item *i_colors = vm->param(0);
   Item *i_first = vm->param(1);

   if ( i_colors == 0 || ! i_colors->isMemBuf() ||
        i_first == 0 || ! i_first->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "M,N" ) ) );
      return;
   }

   MemBuf *colors = i_colors->asMemBuf();
   int first = (int) i_first->forceInteger();
   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();

   vm->retval(
      ::SDL_SetColors( surface, (SDL_Color *) colors->data(), first, colors->length() ) == 0 ?
         false : true );
}

/*#
   @method SetIcon SDLSurface
   @brief Sets this surface as the icon for the SDL window.
   @todo Add the mask parameter and use it.

   This function must be called before the first call to @a SDL.SetVideoMode.
*/
FALCON_FUNC SDLSurface_SetIcon ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();

   ::SDL_WM_SetIcon( surface, NULL );
}



//==================================================================
// Screen class
//

/*#
   @class SDLScreen
   @from SDLSurface
   @brief Screen oriented SDL surface.

   This class is a specialization of the SDLSurface class, providing methods
   meant to be used only on surfaces that should refer to SDL graphic devices.
*/

/*#
   @method UpdateRect SDLScreen
   @brief Refreshes an SDL screen surface or a part of it.
   @optparam xOrRect an X coordinate or a @a SDLRect containing refresh coordinates.
   @optparam y coordinate of the refresh rectangle.
   @optparam width width of the refresh rectangle.
   @optparam height height of the refresh rectangle.

   If no parameter is specified, the whole screen is refreshed. A refresh area can
   be specified either passing a single @a SDLRect instance as parameter or specifying
   the four coordinates as numbers.
*/

FALCON_FUNC SDLScreen_UpdateRect( ::Falcon::VMachine *vm )
{
   // accept a rect as first parameter, or even no parameter.
   CoreObject *self = vm->self().asObject();
   SDL_Surface *screen = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();

   if( vm->paramCount() == 0 )
   {
      ::SDL_UpdateRect( screen, 0, 0, 0, 0);
   }
   else if ( vm->paramCount() == 1 )
   {
      Item *i_rect = vm->param(0);
      SDL_Rect r;

      if( ! i_rect->isObject() || ! ObjectToRect( i_rect->asObject(), r ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "SDLRect|N,[N,N,N]" ) ) );
         return;
      }

      ::SDL_UpdateRect( screen, r.x, r.y, r.w, r.h );
   }
   else
   {
      Item *i_x = vm->param(0);
      Item *i_y = vm->param(1);
      Item *i_w = vm->param(2);
      Item *i_h = vm->param(3);

      if( i_x == 0  || ! i_x->isOrdinal() ||
          i_y == 0  || ! i_y->isOrdinal() ||
          i_w == 0  || ! i_w->isOrdinal() ||
          i_h == 0  || ! i_h->isOrdinal()
        )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "SDLRect|N,[N,N,N]" ) ) );
         return;
      }

      ::SDL_UpdateRect( screen, (Sint32) i_x->forceInteger(), (Sint32) i_y->forceInteger(),
                                (Sint32) i_w->forceInteger(), (Sint32) i_h->forceInteger() );
   }
}

/*#
   @method Flip SDLScreen
   @brief Flips screen buffers.

   It delivers to the screen the paint buffer, be it software or hardware.
*/
FALCON_FUNC SDLScreen_Flip( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDL_Surface *screen = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   ::SDL_Flip( screen );
}


/*#
   @method UpdateRects SDLScreen
   @brief Updates one or more areas of the screen at the same time.
   @param aRects Array of SDLRect instances that have to be updated.

   This method should not be called while helding the lock of the
   screen surface.
*/
FALCON_FUNC SDLScreen_UpdateRects( ::Falcon::VMachine *vm )
{
   Item *i_aRect = vm->param(0);
   if ( i_aRect == 0 || ! i_aRect->isArray() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "A" ) ) );
      return;
   }

   CoreArray *aRect = i_aRect->asArray();
   if( aRect->length() == 0 )
      return;

   SDL_Rect *rects = (SDL_Rect *) memAlloc( sizeof( SDL_Rect ) * aRect->length() );
   for( uint32 i = 0; i = aRect->length(); i++ )
   {
      SDL_Rect *r = rects + i;
      Item &obj = aRect->at( i );
      if ( ! obj.isObject() || ! ObjectToRect( obj.asObject(), *r ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ ).
            extra( "A->SDLRect" ) ) );
         memFree( rects );
         return;
      }
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *screen = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();
   ::SDL_UpdateRects( screen, (int) aRect->length(), rects );
   memFree( rects );
}

/*#
   @method SetPalette SDLScreen
   @brief Changes part of the palette of a palettized surface.
   @param flags Selector of logical or physical palette (or both).
   @param colors A 4 bytes per entry MemBuf containing the color data.
   @param firstColor An integer specifying the first color of the palette.
   @return true if the palette could be changed, 0 if the image has no palette or
           if the color map was too wide to fit the image palette.

   The colors in the MemBuf are stored with red, green, blue stored respectively at
   smallest, medium and highest address (regardless of endianity). In other words,
   the first 8 bits of the color number are the red value, the next 8 bits are the
   green value and the next 8 bits are blue.

   Flags can be one of the following values; if both the palettes (logical and physical)
   must be updated, the two flags can be combined through OR operator.

   - SDL.LOGPAL - updates the logical palette
   - SDL.PHYSPAL - updates the physical palette
*/
FALCON_FUNC SDLScreen_SetPalette( ::Falcon::VMachine *vm )
{
   Item *i_flags = vm->param(0);
   Item *i_colors = vm->param(1);
   Item *i_first = vm->param(2);

   if ( i_flags == 0 || ! i_flags->isInteger() ||
        i_colors == 0 || ! i_colors->isMemBuf() ||
        i_first == 0 || ! i_first->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,M,N" ) ) );
      return;
   }

   MemBuf *colors = i_colors->asMemBuf();
   int flags = (int) i_flags->asInteger();
   int first = (int) i_first->forceInteger();
   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();

   vm->retval(
      ::SDL_SetPalette( surface, flags, (SDL_Color *) colors->data(), first, colors->length() ) == 0 ?
      false : true );
}

/*#
   @method ToggleFullScreen SDLScreen
   @brief Toggles the application between windowed and fullscreen mode
   @raise SDLError if not supported.

   Toggles the application between windowed and fullscreen mode, if supported.
*/
FALCON_FUNC SDLScreen_ToggleFullScreen ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDL_Surface *surface = static_cast<SDLSurfaceCarrier_impl*>( self->getUserData() )->surface();

   if ( SDL_WM_ToggleFullScreen( surface ) == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 11, __LINE__ )
         .desc( "SDL Toggle Full Screen Error" )
         .extra( SDL_GetError() ) ) );
   }

}

//==================================================================
// PixelFormat class
//

/*#
   @class SDLPixelFormat

   This class stores the SDL_PixelFormat structure. Scripts don't usually
   want to use this, but it may be useful to determine and set color
   data for palette based images.

   @prop BitsPerPixel Number of bits per pixel of this image.
   @prop BytesPerPixel Number of bytes per pixel of this image.
   @prop Rloss Red loss factor.
   @prop Gloss Green loss factor.
   @prop Bloss Blue loss factor.
   @prop Aloss Alpha loss factor.

   @prop Rshift Red shift factor.
   @prop Gshift Green shift factor.
   @prop Bshift Blue shift factor.
   @prop Ashift Alpha shift factor.

   @prop Rmask Red bitfield mask.
   @prop Gmask Green bitfield mask.
   @prop Bmask Blue bitfield mask.
   @prop Amask Alpha bitfield mask.
   @prop colorkey Pixel value of transparent pixels.
   @prop alpha Overall image Alpha value.
   @prop palette May be nil or may be an instance of @a SDLPalette if this surface has a palette.
*/

//==================================================================
// Palette class
//

/*#
   @class SDLPalette
   @brief Represents the palette of a surface

   @prop ncolors number of elements in the palette (equal to colors.len() )
   @prop colors MemBuf of 4 byte elements containing each color entry.
*/

/*#
   @method GetColor SDLPalette
   @brief Gets a color in the image palette.
   @param colorIndex Index of the color in the palette.
   @optparam colArray Array of that will hold red, green and blue values.
   @return an array containing red, green and blue elements.
   @raise RangeError if color index is out of range.

   An array can be provided as parameter to prevent re-allocation of the returned value.
*/
FALCON_FUNC  SDLPalette_getColor( ::Falcon::VMachine *vm )
{
   Item *i_index = vm->param( 0 );
   Item *i_target = vm->param( 1 );

   if ( (i_index == 0 || ! i_index->isOrdinal()) ||
        (i_target != 0 && ! i_target->isArray())
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,[A]" ) ) );
         return;
   }

   Item i_colors;
   vm->self().asObject()->getProperty( "colors", i_colors );
   if ( ! i_colors.isMemBuf() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "self.colors.type() != MemBuf" ) ) );
         return;
   }

   MemBuf *colors = i_colors.asMemBuf();

   int64 index = i_index->forceInteger();
   if ( index < 0 || index >= (int64) colors->length() )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_param_range, __LINE__ ) ) );
         return;
   }

   uint32 color = colors->get( (uint32) index );
   CoreArray *array = i_target == 0 ? new CoreArray( vm, 3 ) : i_target->asArray();
   array->append( (int64) (color & 0xff ) ); // red lsb
   array->append( (int64) ((color & 0xff00 ) >> 8 ) ); // green
   array->append( (int64) ((color & 0xff0000 ) >> 16 ) ); // blue
   vm->retval( array );
}

/*#
   @method SetColor SDLPalette
   @brief Sets a color in the image palette.
   @param colorIndex Index of the color in the palette.
   @param red the Red value of the color to be set, or a three elements
      array with the three color values.
   @optparam green Green value of the element (not needed if red was an array).
   @optparam blue Blue value of the element (not needed if red was an array).
   @raise RangeError if color index is out of range.

   Changes a value in the image palette
*/
FALCON_FUNC  SDLPalette_setColor( ::Falcon::VMachine *vm )
{
   Item *i_index = vm->param( 0 );
   Item *i_red = vm->param( 1 );
   Item *i_green = vm->param( 2 );
   Item *i_blue = vm->param( 3 );

   if ( (i_index == 0 || ! i_index->isOrdinal()) ||
        (i_red == 0 || (! i_red->isArray() && ! i_red->isOrdinal() ) )||

        ( i_red->isOrdinal() && (
             (i_green == 0 || ! i_green->isOrdinal()) ||
             (i_blue == 0 || ! i_blue->isOrdinal())
             )
         )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,A|N,[N,N]" ) ) );
         return;
   }

   Item i_colors;
   vm->self().asObject()->getProperty( "colors", i_colors );
   if ( !i_colors.isMemBuf() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "self.colors.type() != MemBuf" ) ) );
      return;
   }

   MemBuf *colors = i_colors.asMemBuf();

   int64 index = i_index->forceInteger();
   if ( index < 0 || index >= (int64) colors->length() )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_param_range, __LINE__ ) ) );
      return;
   }

   // extract the colors
   uint64 red, green, blue;

   if ( i_red->isArray() )
   {
      CoreArray *colArr = i_red->asArray();
      if ( colArr->length() < 3 )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
         return;
      }
      red = colArr->at(0).forceInteger();
      green = colArr->at(1).forceInteger();
      blue = colArr->at(2).forceInteger();
   }
   else
   {
      red = i_red->forceInteger();
      green = i_green->forceInteger();
      blue = i_blue->forceInteger();
   }
   uint32 color = (uint32) ((red &0xff) | ((green & 0xff)<< 8) | ((green & 0xff)<< 16));
   colors->set( (uint32) index, color );
}


}
}

/* end of sdl_surface_ext.cpp */
