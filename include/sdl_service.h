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

#ifndef FALCON_SDL_SERVICE_H
#define FALCON_SDL_SERVICE_H

#include <falcon/setup.h>
#include <falcon/service.h>
#include <falcon/error.h>
#include <falcon/userdata.h>

extern "C" {
   #include <SDL.h>
}


#define FALCON_SDL_ERROR_BASE 2100

namespace Falcon
{

class CoreObject;

/** Base for general SDL exported carrier.

*/
class SDLSurfaceCarrier: public UserData
{
public:
   SDLSurfaceCarrier() {}

   virtual ~SDLSurfaceCarrier() {};
   virtual SDL_Surface* surface() const=0;
};


/**
   Shared SDL module services
*/
class FALCON_DYN_CLASS SDLService: public Service
{

public:
   SDLService();
   virtual CoreObject *createSurfaceInstance( VMachine *vm, ::SDL_Surface *surface );
};


/** Low level SDL error */
class SDLError: public ::Falcon::Error
{
public:
   SDLError():
      Error( "SDLError" )
   {}

   SDLError( const ErrorParam &params  ):
      Error( "SDLError", params )
      {}
};

}

#endif

/* end of sdl_service.h */
