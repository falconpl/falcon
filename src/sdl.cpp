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

   return self;
}

/* end of sdl.cpp */
