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
   @method OpenAudio MIX
   @brief Initialize the MIX module.
   @raise SDLError on initialization failure.

   It is necessary to call @a SDL.Init with SDL.INIT_AUDIO option.
*/

FALCON_FUNC mix_OpenAudio( VMachine *vm )
{
/*
   int retval = ::TTF_Init();
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE, __LINE__ )
         .desc( "Mixer" )
         .extra( TTF_GetError() ) ) );
      return;
   }

   // we can be reasonabily certain that our service is ready here.
   s_service = (SDLService *) vm->getService( "SDLService" );
   if ( s_service == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE+1, __LINE__ )
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
   int retval = ::TTF_Init();
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE, __LINE__ )
         .desc( "Mixer" )
         .extra( TTF_GetError() ) ) );
      return;
   }

   // we can be reasonabily certain that our service is ready here.
   s_service = (SDLService *) vm->getService( "SDLService" );
   if ( s_service == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_TTF_ERROR_BASE+1, __LINE__ )
         .desc( "SDL service not in the target VM" ) ) );
   }
   */
}

}
}

/* end of TTF_ext.cpp */
