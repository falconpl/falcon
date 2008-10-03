/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_falcon.h

   The SDL binding support module - basic inter-module definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 03 Oct 2008 23:41:17 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL binding support module - basic inter-module definitions.
*/

#ifndef FALCON_SDL_FALCON
#define FALCON_SDL_FALCON

#include <falcon/userdata.h>
#include <SDL.h>

namespace Falcon  {

/** Base for general SDL exported carrier.

*/
class SDLSurfaceCarrier: public UserData
{
public:
   SDLSurfaceCarrier() {}

   virtual ~SDLSurfaceCarrier() {};
   virtual SDL_Surface* getSurface() const=0;
};

}

#endif

