/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf.cpp

   The SDL binding support module - True Type extension.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:05:56 +0100
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
   The sdl module - main file.
*/


#include <SDL_ttf.h>

#include <falcon/setup.h>
#include <falcon/enginedata.h>
#include <falcon/module.h>
#include "version.h"
#include "sdlttf_ext.h"

/*#
   @module sdlttf True Type extensions for the Falcon SDL module.
   @brief True Type extensions for the Falcon SDL module.

   This module wraps the True Type extensions for SDL. Namely, this module
   is meant to transform text into graphics that can then be shown on
   @a SDLSurface objects.

   @beginmodule sdlttf
*/


FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "sdlttf" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the main SDL module.
   self->addDepend( self->addString("sdl") );

   //=================================================================
   // Encapsulation SDLTTF
   //

   /*#
      @class SDLTTF
      @brief Main SDL TTF encapsulation class.

      This class is the namespace for TTF functions of the SDL module.
      It contains the extensions provided by Falcon on the sdlttf
      module.
   */

   Falcon::Symbol *c_sdlttf = self->addClass( "SDLTTF" );

   // Init and quit
   self->addClassMethod( c_sdlttf, "Init", Falcon::Ext::ttf_Init );
   self->addClassMethod( c_sdlttf, "WasInit", Falcon::Ext::ttf_WasInit );
   self->addClassMethod( c_sdlttf, "InitAuto", Falcon::Ext::ttf_InitAuto );
   self->addClassMethod( c_sdlttf, "Quit", Falcon::Ext::ttf_Quit );
   self->addClassMethod( c_sdlttf, "Compiled_Version", Falcon::Ext::ttf_Compiled_Version );
   self->addClassMethod( c_sdlttf, "Linked_Version", Falcon::Ext::ttf_Linked_Version );

   //=================================================================
   // Auto quit feature
   //
   Falcon::Symbol *c_sdl_aq = self->addClass( "_SDLTTF_AutoQuit" );
   c_sdl_aq->setWKS( true );
   c_sdl_aq->exported( false );
   self->addClassMethod( c_sdl_aq, "Quit", Falcon::Ext::ttf_Quit );


   return self;
}

/* end of sdlttf.cpp */
