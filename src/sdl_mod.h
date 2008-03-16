/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_mod.h

   The SDL binding support module - module specific extensions.
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
   The SDL binding support module - module specific extensions.
*/

#ifndef FALCON_SDL_MOD
#define FALCON_SDL_MOD

#include <falcon/setup.h>
#include <falcon/userdata.h>
#include <falcon/error.h>

#define FALCON_SDL_ERROR_BASE 2100

namespace Falcon{
namespace Ext{

class QuitCarrier: public UserData
{
public:
   QuitCarrier() {}
   ~QuitCarrier();
};

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
}
#endif

/* end of sdl_mod.h */
