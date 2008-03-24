/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_ext.cpp

   The SDL True Type binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:11:06 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   The SDL True Type binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "sdlttf_ext.h"
#include "sdlttf_mod.h"
#include "../src/sdl_mod.h"  // for SDLError class

#include <SDL_ttf.h>

/*# @beginmodule sdlttf */

namespace Falcon {
namespace Ext {

/*#
   @method Init SDLTTF
   @brief Initialize the TTF module
   @raise SDLError on initialization failure

   Does not require @a SDL.Init to be called before.
*/

FALCON_FUNC ttf_Init( VMachine *vm )
{
   int retval = ::TTF_Init();
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLTTF_ERROR_BASE, __LINE__ )
         .desc( "SDLTTF Error" )
         .extra( TTF_GetError() ) ) );
   }
}

/*#
   @method WasInit SDLTTF
   @brief Detects if the SDLTTF subsystem was initialized.
   @return True if the system was initialized, false otherwise.
*/

FALCON_FUNC ttf_WasInit( VMachine *vm )
{
   vm->retval( (TTF_WasInit() ? true : false ) );
}


/*#
   @method InitAuto SDLTTF
   @brief Initialize the TTF module and prepares for automatic de-initialization.
   @raise SDLError on initialization failure

   Does not require @a SDL.Init to be called before.

   This method returns an object; when the object is destroyed by the GC,
   the library is de-initialized. To perform an application wide intialization
   with automatic de-initialization on quit, just store the return value of
   this method on a global variable of the main script.
*/

FALCON_FUNC ttf_InitAuto( VMachine *vm )
{
   int retval = ::TTF_Init();
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLTTF_ERROR_BASE, __LINE__ )
         .desc( "SDLTTF Init error" )
         .extra( TTF_GetError() ) ) );
      return;
   }

   // also create an object for auto quit.
   Item *c_auto = vm->findWKI( "_SDLTTF_AutoQuit" );
   CoreObject *obj = c_auto->asClass()->createInstance();
   obj->setUserData( new TTFQuitCarrier );
   vm->retval( obj );
}

/*#
   @method Quit SDLTTF
   @brief Turns off SDLTTF system.

   This call shuts down the TTF extensions for SDL and resets system status.
   After a TTF quit, it is possible to reinitialize it.
*/

FALCON_FUNC ttf_Quit( VMachine *vm )
{
   ::TTF_Quit();
}

/*#
   @method Compiled_Version SDLTTF
   @brief Determine the version used to compile this SDL TTF module.
   @return a three element array containing the major, minor and fix versions.
   @see SDLTTF.Linked_Version
*/
FALCON_FUNC ttf_Compiled_Version( VMachine *vm )
{
   SDL_version compile_version;
   TTF_VERSION(&compile_version);

   CoreArray *arr = new CoreArray( vm, 3 );
   arr->append( compile_version.major );
   arr->append( compile_version.minor );
   arr->append( compile_version.patch );
   vm->retval( arr );
}

/*#
   @method Linked_Version SDLTTF
   @brief Determine the version of the library that is currently linked.
   @return a three element array containing the major, minor and fix versions.

   This function determines the version of the SDL_ttf library that is running
   on the system. As long as the interface is the same, it may be different
   from the version used to compile this module.
*/
FALCON_FUNC ttf_Linked_Version( VMachine *vm )
{
   const SDL_version *link_version;
   link_version = TTF_Linked_Version();

   CoreArray *arr = new CoreArray( vm, 3 );
   arr->append( link_version->major );
   arr->append( link_version->minor );
   arr->append( link_version->patch );
   vm->retval( arr );
}

/*#
   @method OpenFont SDLTTF
   @brief Open a font file.
   @return

   This function determines the version of the SDL_ttf library that is running
   on the system. As long as the interface is the same, it may be different
   from the version used to compile this module.
*/
FALCON_FUNC ttf_OpenFont( VMachine *vm )
{

}

}
}

/* end of sdlttf_ext.cpp */
