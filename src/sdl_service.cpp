/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_service.h

   The SDL binding support module - intermodule services.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 25 Mar 2008 02:53:44 +0100
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
   The SDL binding support module - intermodule services.
*/

#include "sdl_service.h"
#include "sdl_mod.h"
#include <falcon/vm.h>

namespace Falcon {

SDLService::SDLService():
   Service( "SDLService" )
{
}

CoreObject *SDLService::createSurfaceInstance( VMachine *vm, ::SDL_Surface *surface )
{
   Item *cls = vm->findWKI( "SDLSurface" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   obj->setUserData( new Ext::SDLSurfaceCarrier( vm, surface ) );
   return obj;
}

}

/* end of sdl_service.cpp */
