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

#ifndef FALCON_SDL_SERVICE_H
#define FALCON_SDL_SERVICE_H

#include <falcon/setup.h>
#include <falcon/service.h>
#include <SDL.h>

namespace Falcon {
class CoreObject;

/**
   Shared SDL module services
*/
class FALCON_DYN_CLASS SDLService: public Service
{

public:
   SDLService();
   virtual CoreObject *createSurfaceInstance( VMachine *vm, ::SDL_Surface *surface );
};

}

#endif

/* end of sdl_service.h */
