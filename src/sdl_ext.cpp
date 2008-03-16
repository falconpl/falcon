/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_ext.cpp

   The SDL binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100
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
   The SDL binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include "sdl_ext.h"
#include "sdl_mod.h"

#include <SDL.h>

namespace Falcon {
namespace Ext {


//=================================================
// Initialization, quit and infos
//

FALCON_FUNC sdl_Init( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   int retval = ::SDL_Init( (Uint32) i_init->forceInteger() );
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Error" )
         .extra( SDL_GetError() ) ) );
   }
}


FALCON_FUNC sdl_InitAuto( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   int retval = ::SDL_Init( (Uint32) i_init->forceInteger() );
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Error" )
         .extra( SDL_GetError() ) ) );
      return;
   }

   // also create an object for auto quit.
   Item *c_auto = vm->findWKI( "_SDL_AutoQuit" );
   CoreObject *obj = c_auto->asClass()->createInstance();
   obj->setUserData( new QuitCarrier );
   vm->retval( obj );
}


FALCON_FUNC sdl_WasInit( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   Uint32 mask = ::SDL_WasInit( (Uint32) i_init->forceInteger() );
   vm->retval( (int64) mask );
}


FALCON_FUNC sdl_Quit( ::Falcon::VMachine *vm )
{
   SDL_Quit();
}

FALCON_FUNC sdl_QuitSubSystem( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   SDL_QuitSubSystem( (Uint32) i_init->forceInteger() );
}

//==================================================================
// ERROR class

FALCON_FUNC  SDLError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new SDLError ) );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of sdl_ext.cpp */
