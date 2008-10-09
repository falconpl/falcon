/*
   FALCON - The Falcon Programming Language.
   FILE: sdlimage.cpp

   The SDL binding support module - image loading extension.
   -------------------------------------------------------------------
   Author: Federico Baroni
   Begin: Wed, 24 Sep 2008 23:04:56 +0100

   Last modified because:
   Tue 7 Oct 2008 23:06:03 - GetError and SetError added

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The sdl module - main file.
*/


#include <falcon/setup.h>
#include <falcon/enginedata.h>
#include <falcon/module.h>
#include "version.h"
#include "sdlimage_ext.h"

/*#
   @module sdlimage image loading extensions for the Falcon SDL module.
   @brief Image loading extensions for the Falcon SDL module.

   This module wraps the image loading extensions for SDL. Namely, this module
   is meant to load images from files that can be shown on
   @a SDLSurface objects.

   @beginmodule sdlimage
*/


FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name ( "sdlimage" );
   self->language( "en_US" );
   self->engineVersion ( FALCON_VERSION_NUM );
   self->version ( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the main SDL module.
   self->addDepend( "sdl" );

   //=================================================================
   // Encapsulation SDLIMAGE
   //

   /*#
      @class IMAGE
      @brief Main SDL IMAGE encapsulation class.

      This class is the namespace for IMAGE functions of the SDL module.
      It contains the extensions provided by Falcon on the sdlimage
      module.
   */

   Falcon::Symbol *c_sdlimage = self->addClass ( "IMAGE" );

   // No specific interface defines, so no addClassProperty required
   // i.e.: self->addClassProperty( c_sdlttf, "STYLE_BOLD" ).setInteger( TTF_STYLE_BOLD );

   // Interface methods

   // Loading
   self->addClassMethod( c_sdlimage, "Load", Falcon::Ext::img_Load ).asSymbol()->
        addParam ( "filename" );
   self->addClassMethod( c_sdlimage, "LoadRW", Falcon::Ext::img_LoadRW ).asSymbol()->
        addParam ( "data" );

   // Info
   self->addClassMethod( c_sdlimage, "IsJPG", Falcon::Ext::img_isJPG ).asSymbol()->
        addParam ( "src" );

   // Error
   self->addClassMethod( c_sdlimage, "GetError", Falcon::Ext::img_GetError );
   self->addClassMethod( c_sdlimage, "SetError", Falcon::Ext::img_SetError ).asSymbol()->
        addParam ( "error_str" );


   return self;
}

/* end of sdlimage.cpp */

