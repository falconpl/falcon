/*
   FALCON - The Falcon Programming Language.
   FILE: sdlimage_ext.cpp

   The SDL image loading binding support module.
   -------------------------------------------------------------------
   Author: Federico Baroni
   Begin: Tue, 30 Sep 2008 23:05:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL image loading binding support module.
*/

#include <falcon/vm.h>
//#include <falcon/transcoding.h>
//#include <falcon/fstream.h>
//#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
//#include <falcon/membuf.h>

#include <sdl_service.h>

#include "sdlimage_ext.h"

extern "C" {
   #include <SDL_image.h>
}

//#include <iostream> // Da rimuovere


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
         .extra( SDL_GetError() ) ) );
      return;
   }

   // Copy the new item in a surface
   SDLService *sdlservice = (SDLService*)vm->getService ( "SDLService" );
   CoreObject* ret = sdlservice->createSurfaceInstance ( vm, surf );

   vm->retval( ret );

}

/*#
   @method img_isJPG
   @brief Checks if the image is a JPG file
   @return The return value is 1 if the image is a JPG file, 0 if it's not a JPG encoded file

   the image data is tested to see if it is readable as a JPG, otherwise it returns false (Zero).
*/

FALCON_FUNC img_isJPG ( VMachine *vm )
{
   // Something here
}

/*#
   @method img_GetError
   @brief fff
   @return fff

   adkjaòldkfj
*/

FALCON_FUNC img_GetError ( VMachine *vm )
{
   // Something here
}

}
}

/* end of sdlimage_ext.cpp */
