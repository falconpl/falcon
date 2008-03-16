/*
   FALCON - The Falcon Programming Language.
   FILE: sdl.cpp

   The SDL binding support module.
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
   The sdl module - main file.
*/

extern "C" {
   #include <SDL.h>
}

#include <falcon/setup.h>
#include <falcon/enginedata.h>
#include <falcon/module.h>
#include "version.h"
#include "sdl_ext.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "sdl" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //=================================================================
   // Encapsulation SDL
   //
   Falcon::Symbol *c_sdl = self->addClass( "SDL" );
   self->addClassProperty( c_sdl, "INIT_VIDEO" )->setInteger( SDL_INIT_VIDEO );
   self->addClassProperty( c_sdl, "INIT_AUDIO" )->setInteger( SDL_INIT_AUDIO );
   self->addClassProperty( c_sdl, "INIT_TIMER" )->setInteger( SDL_INIT_TIMER );
   self->addClassProperty( c_sdl, "INIT_CDROM" )->setInteger( SDL_INIT_CDROM );
   self->addClassProperty( c_sdl, "INIT_JOYSTICK" )->setInteger( SDL_INIT_JOYSTICK );
   self->addClassProperty( c_sdl, "INIT_EVERYTHING" )->setInteger( SDL_INIT_EVERYTHING );
   self->addClassProperty( c_sdl, "INIT_NOPARACHUTE" )->setInteger( SDL_INIT_NOPARACHUTE );

   self->addClassMethod( c_sdl, "Init", Falcon::Ext::sdl_Init );
   self->addClassMethod( c_sdl, "WasInit", Falcon::Ext::sdl_WasInit );
   self->addClassMethod( c_sdl, "InitAuto", Falcon::Ext::sdl_InitAuto );
   self->addClassMethod( c_sdl, "Quit", Falcon::Ext::sdl_Quit );
   self->addClassMethod( c_sdl, "QuitSubSystem", Falcon::Ext::sdl_QuitSubSystem );

   //============================================================
   // SDL Error class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *sdlerr_cls = self->addClass( "SDLError", Falcon::Ext::SDLError_init );
   sdlerr_cls->setWKS( true );
   sdlerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   //=================================================================
   // Auto quit feature
   //
   Falcon::Symbol *c_sdl_aq = self->addClass( "_SDL_AutoQuit" );
   c_sdl_aq->setWKS( true );
   c_sdl_aq->exported( false );
   self->addClassMethod( c_sdl_aq, "Quit", Falcon::Ext::sdl_Quit );

   return self;
}

/* end of sdl.cpp */
