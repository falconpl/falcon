/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_service.h

   The SDL binding support module - intermodule services.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 25 Mar 2008 02:53:44 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL binding support module - intermodule services.
*/

#define FALCON_EXPORT_SERVICE

#include "sdl_service.h"
#include "sdl_mod.h"
#include <falcon/vm.h>

namespace Falcon {

SDLService::SDLService():
   Service( "SDLService" )
{
}

SDLService::~SDLService()
{
}

CoreObject *SDLService::createSurfaceInstance( VMachine *vm, ::SDL_Surface *surface )
{
   Item *cls = vm->findWKI( "SDLSurface" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   obj->setUserData( new Ext::SDLSurfaceCarrier_impl( surface ) );
   return obj;
}

}

/* end of sdl_service.cpp */
