/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_cursor_ext.cpp

   The SDL binding support module - cursor extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 22 Mar 2008 17:05:49 +0100

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

/*#
   @beginmodule sdl
*/

namespace Falcon {
namespace Ext {

/*#
   @method GetCursor SDL
   @brief Gets currently active cursor.
   @return an instance of @a SDLCursor class or nil.

   This method returns the currently active cursor. It can be used to
   store this away and set it later.
*/
FALCON_FUNC sdl_GetCursor( ::Falcon::VMachine *vm )
{
   ::SDL_Cursor *cursor = ::SDL_GetCursor();
   if ( cursor == 0 )
      vm->retnil();
   else {
      Item *cls = vm->findWKI( "SDLCursor" );
      fassert( cls != 0 );
      CoreObject *obj = cls->asClass()->createInstance();
      obj->setUserData( new SDLCursorCarrier( cursor, false ) );
      vm->retval( obj );
   }
}

/*#
   @method ShowCursor SDL
   @optparam request The request for the cursor.
   @brief Changes or query the visibility of the mouse cursor.
   @return Current status of cursor visibility.

   Toggle whether or not the cursor is shown on the screen.
   Passing SDL.ENABLE displays the cursor and passing SDL.DISABLE hides it.
   The current state of the mouse cursor can be queried by passing SDL.QUERY,
   either SDL.DISABLE or SDL.ENABLE will be returned.

   The cursor starts off displayed, but can be turned off.

   If the request parameter is not given, it defaults to SDL.ENABLE.
*/
FALCON_FUNC sdl_ShowCursor( ::Falcon::VMachine *vm )
{
   int mode;
   Item *param;

   if ( vm->paramCount() == 0 )
   {
      mode = SDL_ENABLE;
   }
   else if ( ! ( param = vm->param(0) )->isInteger() ||
         ( ( mode = param->asInteger() ) != SDL_ENABLE &&
            mode != SDL_DISABLE &&
            mode != SDL_QUERY )
      )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "I" ) ) ;
         return;
      }


   vm->retval( (int64) ::SDL_ShowCursor( mode ) );
}

/*#
   @method MakeCursor SDL
   @brief Builds an SDL cursor taking an array of strings as input.
   @param aImage a string image.
   @param hotX X coordinate of the hotspot.
   @param hotY Y coordinate of the hotspot.

   As it is generally quite hard to build a cursor writing directly
   its binary data, this helper method takes a set of strings in
   input and convers their visual representation to a valid cursor.

   The strings must have a lenght mutliple of 8, as each of its
   element will fill a bit in the final cursor.

   The hotspot coordinates must be equal to or smaller than the width and the
   height of the cursor. The width of the cursor is determined by the width
   of the strings, while its height is determined the size of the vector.

   The characters in the image are considered as follows:

      - "\@": white opaque pixel
      - ".": black opaque pixel
      - "X": reverse pixel (not always available)
      - " ": transparent pixel.

   In example, the following code generates a small cross with white borders, a reverse
   inner part and a small black shadow:
   @code
      strImage = [
         "         @XXX@          ",
         "         @XXX@          ",
         "         @XXX@          ",
         "         @X.X@          ",
         "  @@@@@@@@X.X@@@@@@@@   ",
         "  XXXXXXXXX.XXXXXXXXX.  ",
         "  XXXXXXXXX.XXXXXXXXX.  ",
         "  @@@@@@@@X.X@@@@@@@@.  ",
         "         @X.X@........  ",
         "         @XXX@.         ",
         "         @XXX@.         ",
         "         @XXX@.         ",
         "          ....          " ]

      SDL.MakeCursor( strImage, 12, 7 ).SetCursor()
   @endcode
*/
FALCON_FUNC sdl_MakeCursor( ::Falcon::VMachine *vm )
{
   Uint8 *data = 0, *mask = 0;
   Item *i_array, *i_xspot, *i_yspot;

   if ( vm->paramCount() < 3 ||
      ( ! (i_array = vm->param(0))->isArray() ) ||
      ( ! (i_xspot = vm->param(1))->isOrdinal() ) ||
      ( ! (i_yspot = vm->param(2))->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "A,N,N" ) ) ;
      return;
   }

   int xspot = i_xspot->forceInteger();
   int yspot = i_yspot->forceInteger();

   CoreArray *array = i_array->asArray();
   int height = (int) array->length();
   if ( height < 1 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Array empty" ) ) ;
      return;
   }

   int width = -1;

   for ( int i = 0; i < height; i ++ )
   {
      Item &elem = array->at(i);
      if( ! elem.isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "Array contains non-strings" ) ) ;
         return;
      }

      String &row = *elem.asString();
      if( row.length() == 0 || row.length() % 8 != 0 )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "Strings not modulo 8" ) ) ;
         if ( data != 0 )
         {
            free( data );
            free( mask );
         }
         return;
      }

      if( width == -1 )
      {
         // calculate first width
         width = row.length();
         data = (Uint8 *) malloc( width / 8 * height );
         mask = (Uint8 *) malloc( width / 8 * height );
         memset( data, 0, width / 8 * height );
         memset( mask, 0, width / 8 * height );
      }
      else if ( width != (int) row.length() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "Strings of different sizes" ) ) ;
         return;
      }

      // perform the parsing
      Uint8 *pdata = data + i * width/8;
      Uint8 *pmask = mask + i * width/8;

      for ( int j = 0; j < width; j ++ )
      {
         uint32 chr = row.getCharAt( j );

         switch( chr )
         {
            case ' ': // nothing to do 0/0
               break;

            case 'X': // inverse 1/0
               pdata[ j/8 ] |= 1 << (7 - (j % 8));
               break;

            case '@':  // white 0/1
               pmask[ j/8 ] |= 1 << (7 -(j % 8));
               break;

            case '.': // black 1/1
               pdata[ j/8 ] |= 1 << (7 -(j % 8));
               pmask[ j/8 ] |= 1 << (7 -(j % 8));
               break;

            default:
               // error
               free( data );
               free( mask );
               throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
                  extra( "Unrecognized char in string" ) );
               return;
         }
      }
   }

   ::SDL_Cursor *cursor = ::SDL_CreateCursor( data, mask, width, height, xspot, yspot );
   Item *cls = vm->findWKI( "SDLCursor" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   obj->setUserData( new SDLCursorCarrier( cursor, true ) );

   free( data );
   free( mask );

   vm->retval( obj );
}

/*#
   @method CreateCursor SDL
   @brief Gets currently active cursor.
   @param mbData MemBuf containing visual bit data.
   @param mbMask Membuf containing visibility mask data.
   @param width Width of the cursor.
   @param height Height of the cursor.
   @param Xspot X position of the cursor hotspot.
   @param Yspot Y position of the cursor hotspot.
   @raise SDLError if the cursor couldn't be created.

   See SDL_CreateCursor documentation. Method @a SDL.MakeCursor is
   probably simpler to use.
*/
FALCON_FUNC sdl_CreateCursor( ::Falcon::VMachine *vm )
{
   Item *i_data, *i_mask;
   Item *i_width, *i_height, *i_xspot, *i_yspot;

   if( vm->paramCount() < 6 ||
      ! (i_data = vm->param(0) )->isMemBuf() ||
      ! (i_mask = vm->param(1) )->isMemBuf() ||
      ! (i_width = vm->param(2) )->isOrdinal() ||
      ! (i_height = vm->param(3) )->isOrdinal() ||
      ! (i_xspot = vm->param(4) )->isOrdinal() ||
      ! (i_yspot = vm->param(5) )->isOrdinal()
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "M,M,N,N,N,N" ) ) ;
      return;
   }

   MemBuf *data = i_data->asMemBuf();
   MemBuf *mask = i_mask->asMemBuf();
   // we are not interested in their word size.
   if( data->size() || data->size() != mask->size() )
   {
      throw new ParamError( ErrorParam( e_param_type, __LINE__ ).
         extra( "Membuf must be of same size" ) ) ;
      return;
   }

   int width = (int) i_width->forceInteger();
   int height = (int) i_height->forceInteger();
   int xspot = (int) i_xspot->forceInteger();
   int yspot = (int) i_yspot->forceInteger();

   if( width < 8 || height < 1 || width % 8 != 0 )
   {
      throw new ParamError( ErrorParam( e_param_type, __LINE__ ).
         extra( "Invalid sizes" ) ) ;
      return;
   }

   if( data->size() != (uint32)( width/8 * height ) )
   {
      throw new ParamError( ErrorParam( e_param_type, __LINE__ ).
         extra( "Membuf doesn't match width and height" ) ) ;
      return;
   }

   if( xspot < 0  || xspot >= width || yspot < 0 || yspot >= height )
   {
      throw new ParamError( ErrorParam( e_param_type, __LINE__ ).
         extra( "Hotspot outside cursor" ) ) ;
      return;
   }

   // ok, all fine
   ::SDL_Cursor *cursor = SDL_CreateCursor( (Uint8 *) data->data(), (Uint8 *) mask->data(),
      width, height, xspot, yspot );

   if ( cursor == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 11, __LINE__ )
         .desc( "SDL Create Cursor" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   Item *cls = vm->findWKI( "SDLCursor" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   obj->setUserData( new SDLCursorCarrier( cursor, true ) );
   vm->retval( obj );
}

/*#
   @class SDLCursor
   @brief Cursor representation for SDL.

   This class holds an opaque object that is used by SDL as a cursor definition.

   Cursors created by the user (i.e. through its class constructor) are automatically
   freed with SDL_FreeCursor when the garbage collector rips the owning object.
*/


/*#
   @method SetCursor SDLCursor
   @brief Selects the current cursor for the SDL.
*/
FALCON_FUNC SDLCursor_SetCursor( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   SDL_Cursor *source = static_cast<SDLCursorCarrier*>( self->getUserData() )->m_cursor;
   SDL_SetCursor( source );
}

}
}

/* end of sdl_curosr_ext.cpp */
