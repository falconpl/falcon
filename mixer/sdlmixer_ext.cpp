/*
   FALCON - The Falcon Programming Language.
   FILE: sdlmixer_ext.cpp

   The SDL Mixer binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Oct 2008 19:11:27 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL Mixer binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "sdlmixer_ext.h"
#include "sdlmixer_mod.h"
#include <sdl_service.h>  // for the reset

extern "C" {
   #include <SDL_mixer.h>
}


/*# @beginmodule sdlmixer */

namespace Falcon {

static SDLService *s_service = 0;

namespace Ext {

/*#
   @method Compiled_Version MIX
//    @brief Determine the version used to compile this SDL mixer module.
   @return a three element array containing the major, minor and fix versions.
   @see MIX.Linked_Version
*/
FALCON_FUNC mix_Compiled_Version( VMachine *vm )
{
   SDL_version compile_version;
   MIX_VERSION(&compile_version);

   CoreArray *arr = new CoreArray( vm, 3 );
   arr->append( (int64) compile_version.major );
   arr->append( (int64) compile_version.minor );
   arr->append( (int64) compile_version.patch );
   vm->retval( arr );
}

/*#
   @method Linked_Version MIX
   @brief Determine the version of the library that is currently linked.
   @return a three element array containing the major, minor and fix versions.

   This function determines the version of the SDL_mixer library that is running
   on the system. As long as the interface is the same, it may be different
   from the version used to compile this module.
*/
FALCON_FUNC mix_Linked_Version( VMachine *vm )
{
   const SDL_version *link_version;
   link_version = Mix_Linked_Version();

   CoreArray *arr = new CoreArray( vm, 3 );
   arr->append( (int64) link_version->major );
   arr->append( (int64) link_version->minor );
   arr->append( (int64) link_version->patch );
   vm->retval( arr );
}


/*#
   @method OpenAudio MIX
   @brief Initialize the MIX module.
   @raise SDLError on initialization failure.

   It is necessary to call @a SDL.Init with SDL.INIT_AUDIO option.
*/

FALCON_FUNC mix_OpenAudio( VMachine *vm )
{
/*
   if ( i_stream == 0 || ! i_stream->isObject() || ! i_stream->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "Stream" ) ) );
      return;
   }

   int retval = ::MIX_OpenAudio();
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE, __LINE__ )
         .desc( "Mixer" )
         .extra( MIX_GetError() ) ) );
      return;
   }

   // we can be reasonabily certain that our service is ready here.
   s_service = (SDLService *) vm->getService( "SDLService" );
   if ( s_service == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_MIX_ERROR_BASE+1, __LINE__ )
         .desc( "SDL service not in the target VM" ) ) );
   }
   */
}

/*#
   @method CloseAudio MIX
   @brief Turn off the MIX module.
   @raise SDLError on initialization failure.

   This method turns off he mixing facility. It must becalled
   the same number of times @a MIX.OpenAudio has been called.
*/

FALCON_FUNC mix_CloseAudio( VMachine *vm )
{
/*
   int retval = ::MIX_Init();
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE, __LINE__ )
         .desc( "Mixer" )
         .extra( MIX_GetError() ) ) );
      return;
   }

   // we can be reasonabily certain that our service is ready here.
   s_service = (SDLService *) vm->getService( "SDLService" );
   if ( s_service == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_MIX_ERROR_BASE+1, __LINE__ )
         .desc( "SDL service not in the target VM" ) ) );
   }
   */
}

}
}

/* end of MIX_ext.cpp */
