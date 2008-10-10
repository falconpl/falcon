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

/*#
   @method img_Load
   @brief Loads an image file based on filename extension
   @return It returns a pointer to the image as a new SDL_Surface. NULL is returned on errors, such as no support built for the image, or a file reading error.

   Load file for use as an image in a new surface. This actually uses the file extension as the type string. This can load all supported image files, provided that the extension is correctly representing the file type.
*/

FALCON_FUNC img_Load ( VMachine *vm )
{
   // Check provided filename
   Item *i_file = vm->param(0);
   if ( i_file == 0 || ! i_file->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Not a filename" ) ) );
      return;
   }

   // Convert filename to a C string
   AutoCString fname( *i_file->asString() );

   // Load the new image
   ::SDL_Surface *surf = ::IMG_Load( fname.c_str() );
   if ( surf == NULL )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 3, __LINE__ )
         .desc( "IMG_Load" )
         .extra( IMG_GetError() ) ) );
      return;
   }

   // Copy the new item in a surface
   SDLService *sdlservice = (SDLService*)vm->getService ( "SDLService" );
   CoreObject* ret = sdlservice->createSurfaceInstance ( vm, surf );

   vm->retval( ret );

}

/*#
   @method img_LoadRW
   @brief Loads an image file already available in a RWops
   @return It returns a pointer to the image as a new SDL_Surface. NULL is returned on errors, such as no support built for the image or some other error.

   Loads data for use as an image in a new surface. It do not support TGA images!
*/

FALCON_FUNC img_LoadRW ( VMachine *vm )
{
   // Check provided filename
   //Item *i_rwops = vm->param(0);
   //Item *i_free = vm->param(1);
   
}

/*#
   @method img_isJPG
   @brief fff
   @return fff

   adkjaòldkfj
*/

FALCON_FUNC img_isJPG ( VMachine *vm )
{
   // Something here
}

/*#
   @method img_GetError
   @brief Gets image related error
   @return Returns a string containing a humam readble version or the reason for the last error that occured

   The current active image error is returned
*/

FALCON_FUNC img_GetError ( VMachine *vm )
{
   // Returns the available error
   vm->retval ( IMG_GetError () );
}

/*#
   @method img_SetError
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Not a string" ) ) );
      return;
   }

   // Convert i_string to a C string
   AutoCString serror( *i_string->asString() );

   // Setting the new error string
   IMG_SetError ( serror.c_str () );
}


}
}

/* end of sdlimage_ext.cpp */
