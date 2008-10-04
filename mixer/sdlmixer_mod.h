/*
   FALCON - The Falcon Programming Language.
   FILE: sdlmixer_mod.h

   The SDL Mixer binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Oct 2008 19:11:27 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL Mixer binding support module - module specific extensions.
*/

#ifndef FALCON_SDLMIXER_MOD
#define FALCON_SDLMIXER_MOD

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include <falcon/error.h>

extern "C" {
   #include <SDL_mixer.h>
}

#ifndef FALCON_SDLMIXER_ERROR_BASE
#define FALCON_SDLMIXER_ERROR_BASE 2140
#endif

#endif

/* end of sdlttf_mod.h */
