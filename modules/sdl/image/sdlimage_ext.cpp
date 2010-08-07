/*
   FALCON - The Falcon Programming Language.
   FILE: sdlimage_ext.cpp

   The SDL image loading binding support module.
   -------------------------------------------------------------------
   Author: Federico Baroni
   Begin: Tue, 30 Sep 2008 23:05:06 +0100

   Last modified because:
   Tue 7 Oct 2008 23:06:03 - GetError and SetError added

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL image loading binding support module.
*/

#include <falcon/vm.h>
#include <falcon/autocstring.h>

#include <sdl_service.h>

#include "sdlimage_ext.h"

extern "C" {
   #include <SDL_image.h>
}

/*# @beginmodule sdlimage */

namespace Falcon {
namespace Ext {

// Cached instance of our service
static SDLService *s_sdlservice = 0;

/*#
   @method Load IMAGE
   @brief Loads an image file based on filename extension
   @param file The name of the file to be loaded, or a Falcon Stream
          instance pointing at the beginning of the image data.
   @optparam type A string representing the type of image in the @b file stream.
   @return A new instance of @a SDLSurface.
   @raise SDLError if the file could not be interpreted as an image.

   Load file for use as an image in a new surface.
   This actually uses the file extension as the type string.
   This can load all supported image files,
   provided that the extension is correctly representing the file type.

   When a stream is given as @b file parameter, the @b type parameter is
   taken as the format type of the image in the incoming stream. It is just
   an hint, some common format doesn't require necessarily it.

   Currently, the
   only format which is not automatically recognized is "TGA"; however, providing
   the hint will result in less guessing and may be necessary if the underlying
   stream doesn't support seek (i.e. a network stream or the standard input).

   Here is a list of the currently recognized strings (case is not important):
      - "TGA"
      - "BMP"
      - "PNM"
      - "XPM"
      - "XCF"
      - "PCX"
      - "GIF"
      - "JPG"
      - "TIF"
      - "LBM"
      - "PNG"
*/

FALCON_FUNC img_Load ( VMachine *vm )
{
   // Check provided parameters
   Item *i_file = vm->param(0);
   Item *i_type = vm->param(1);

   if ( i_file == 0 ||
      ( ! i_file->isString() &&
            !( i_file->isObject() && i_file->asObject()->derivedFrom("Stream") ))||
      ( i_type != 0 && ! i_type->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S|Stream, [S]" ) ) ;
      return;
   }

   // see if we need to cache our service
   if( s_sdlservice == 0 )
   {
      s_sdlservice = static_cast<SDLService*>(vm->getService ( SDL_SERVICE_SIGNATURE ));
   }

   // Load the new image
   ::SDL_Surface *surf;
   if( i_file->isString() )
   {
      // Convert filename to a C string
      AutoCString fname( *i_file->asString() );

      surf = ::IMG_Load( fname.c_str() );
      if ( surf == NULL )
      {
         throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 3, __LINE__ )
            .desc( "IMG_Load" )
            .extra( IMG_GetError() ) ) ;
         return;
      }
   }
   else {
      struct SDL_RWops rwops;
      Stream* stream = static_cast<Stream *>(i_file->asObject()->getUserData());
      s_sdlservice->rwopsFromStream( rwops, stream );
      if ( i_type != 0 )
      {
         AutoCString type( *i_type->asString() );
         surf = ::IMG_LoadTyped_RW( &rwops, 0, const_cast<char *>( type.c_str() ) );
      }
      else
         surf = ::IMG_Load_RW( &rwops, 0 );
   }

   // Copy the new item in a surface
   CoreObject* ret = s_sdlservice->createSurfaceInstance ( vm, surf );

   vm->retval( ret );
}


//===============================================
// Informative functions
// As the are all alike, we use a single check
// configured by the front-end callers.
//===============================================

typedef int (*t_check_func)(SDL_RWops *);

static void img_checkImageType( VMachine *vm, t_check_func func )
{
   // see if we need to cache our service
   if( s_sdlservice == 0 )
   {
      s_sdlservice = static_cast<SDLService*>(vm->getService ( SDL_SERVICE_SIGNATURE ));
   }

   // Check provided filename
   Item *i_file = vm->param(0);
   if ( i_file == 0 ||
        !( i_file->isObject() && i_file->asObject()->derivedFrom("Stream") )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Stream" ) ) ;
      return;
   }

   // Prepare the rwops
   struct SDL_RWops rwops;
   Stream* stream = static_cast<Stream *>(i_file->asObject()->getUserData());
   s_sdlservice->rwopsFromStream( rwops, stream );

   // perform the check and return the value.
   vm->regA().setBoolean( func( &rwops ) != 0 );
}

/*#
   @method isBMP IMAGE
   @brief Checks if an image is in BMP format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a BMP image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/

FALCON_FUNC img_isBMP ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isBMP );
}

/*#
   @method isGIF IMAGE
   @brief Checks if an image is in GIF format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a GIF image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isGIF ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isGIF );
}

/*#
   @method isJPG IMAGE
   @brief Checks if an image is in JPG format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a JPG image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isJPG ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isJPG );
}

/*#
   @method isLBM IMAGE
   @brief Checks if an image is in LBM format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a LBM image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isLBM ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isLBM );
}

/*#
   @method isPCX IMAGE
   @brief Checks if an image is in PCX format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a PCX image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isPCX ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isPCX );
}

/*#
   @method isPNG IMAGE
   @brief Checks if an image is in PCX format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a PCX image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isPNG ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isPNG );
}

/*#
   @method isPNM IMAGE
   @brief Checks if an image is in PNM format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a PNM image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isPNM ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isPNM );
}

/*#
   @method isTIF IMAGE
   @brief Checks if an image is in TIF format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a TIF image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isTIF ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isTIF );
}

/*#
   @method isXCF IMAGE
   @brief Checks if an image is in XCF format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a XCF image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isXCF ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isXCF );
}

/*#
   @method isXPM IMAGE
   @brief Checks if an image is in XPM format.
   @param file a Falcon Stream instance pointing at the beginning of the image data.
   @return True if the stream is stroing a XPM image.

   It may be useful to reset the stream to the current position after calling
   this function.
*/
FALCON_FUNC img_isXPM ( VMachine *vm )
{
   img_checkImageType( vm, ::IMG_isXPM );
}

/*#
   @method GetError IMAGE
   @brief Gets image related error
   @return Returns a string containing a humam readble version or the reason for the last error that occured

   The current active image error is returned
*/

FALCON_FUNC img_GetError ( VMachine *vm )
{
   // Returns the available error
   vm->retval( new CoreString(IMG_GetError ()) );
}

/*#
   @method SetError IMAGE
   @brief Sets image related error string
   @return Returns a string containing a humam readble version or the reason for the last error that occured

   This function sets the error string which may be fetched with img_GetError (or sdl_GetError). The function accepts a string not longer than 1024 chars in length.
*/

FALCON_FUNC img_SetError ( VMachine *vm )
{
   // Check error string
   Item *i_string = vm->param(0);

   // Is a string?
   if ( i_string == 0 || ! i_string->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Not a string" ) ) ;
      return;
   }

   // Convert i_string to a C string
   AutoCString serror( *i_string->asString() );

   // Setting the new error string
   ::IMG_SetError ( serror.c_str() );
}


}
}

/* end of sdlimage_ext.cpp */
