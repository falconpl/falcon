/*
   FALCON - The Falcon Programming Language.
   FILE: sdlmixer.cpp

   The SDL binding support module - Mixer extension.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Oct 2008 19:11:27 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The sdl MIXER module - main file.
*/

extern "C"
{
   #include <SDL_mixer.h>
}

#include <falcon/setup.h>
#include <falcon/enginedata.h>
#include <falcon/module.h>
#include "version.h"
#include "sdlmixer_ext.h"
#include "sdlmixer_mod.h"

/*#
   @module sdlmixer SDL Audio Mixer module for The Falcon Programming Language.
   @brief SDL AUDIO extensions for the Falcon SDL module.

   @beginmodule sdlmixer
*/


FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "sdlmixer" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the main SDL module.
   self->addDepend( "sdl" );

   //=================================================================
   // Encapsulation SDLMIXER
   //

   /*#
      @class MIX
      @brief Main SDL Mixer encapsulation class.

      This class is the namespace for MIX functions of the SDL module.
      It contains the extensions provided by Falcon on the sdlmix
      module.
   */

   Falcon::Symbol *c_sdlmix = self->addClass( "MIX" );
   // Init and quit
   self->addClassMethod( c_sdlmix, "OpenAudio", Falcon::Ext::mix_OpenAudio );
   self->addClassMethod( c_sdlmix, "CloseAudio", Falcon::Ext::mix_CloseAudio );

   return self;
}

/* end of sdlmixer.cpp */

