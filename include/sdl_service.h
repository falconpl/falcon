/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_service.h

   The SDL binding support module - intermodule services.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Oct 2008 19:11:27 +0200

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
#define FALCON_SDL_USER_EVENT (SDL_NUMEVENTS-1)
#define FALCON_SDL_CHANNEL_DONE_EVENT (SDL_NUMEVENTS-2)
#define FALCON_SDL_MUSIC_DONE_EVENT (SDL_NUMEVENTS-3)

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
class FALCON_SERVICE SDLService: public Service
{

public:
   SDLService();
   virtual ~SDLService();
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
