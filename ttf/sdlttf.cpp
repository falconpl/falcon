/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf.cpp

   The SDL binding support module - True Type extension.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:05:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The sdl module - main file.
*/

extern "C"
{
   #include <SDL_ttf.h>
}

#include <falcon/setup.h>
#include <falcon/enginedata.h>
#include <falcon/module.h>
#include "version.h"
#include "sdlttf_ext.h"
#include "sdlttf_mod.h"

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
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // first of all, we need to declare our dependency from the main SDL module.
   self->addDepend( "fsdl" );

   //=================================================================
   // Encapsulation SDLTTF
   //

   /*#
      @class TTF
      @brief Main SDL TTF encapsulation class.

      This class is the namespace for TTF functions of the SDL module.
      It contains the extensions provided by Falcon on the sdlttf
      module.
   */

   Falcon::Symbol *c_sdlttf = self->addClass( "TTF" );
   self->addClassProperty( c_sdlttf, "STYLE_BOLD" ).setInteger( TTF_STYLE_BOLD );
   self->addClassProperty( c_sdlttf, "STYLE_ITALIC" ).setInteger( TTF_STYLE_ITALIC );
   self->addClassProperty( c_sdlttf, "STYLE_UNDERLINE" ).setInteger( TTF_STYLE_UNDERLINE );
   self->addClassProperty( c_sdlttf, "STYLE_NORMAL" ).setInteger( TTF_STYLE_NORMAL );

   // Init and quit
   self->addClassMethod( c_sdlttf, "Init", Falcon::Ext::ttf_Init );
   self->addClassMethod( c_sdlttf, "WasInit", Falcon::Ext::ttf_WasInit );
   self->addClassMethod( c_sdlttf, "InitAuto", Falcon::Ext::ttf_InitAuto );
   self->addClassMethod( c_sdlttf, "Quit", Falcon::Ext::ttf_Quit );
   self->addClassMethod( c_sdlttf, "Compiled_Version", Falcon::Ext::ttf_Compiled_Version );
   self->addClassMethod( c_sdlttf, "Linked_Version", Falcon::Ext::ttf_Linked_Version );
   self->addClassMethod( c_sdlttf, "OpenFont", Falcon::Ext::ttf_OpenFont ).asSymbol()->
      addParam("fontname")->addParam("ptsize")->addParam("index");
   self->addClassMethod( c_sdlttf, "ByteSwappedUNICODE", Falcon::Ext::ttf_ByteSwappedUNICODE ).asSymbol()->
      addParam("swap");


   //=================================================================
   // TTFFont class
   //

   /*#
      @class TTFFont
      @brief Font representation of SDL TTF Class.

      This class encapsulates all the font-specific methods and
      accessros that are provided by th SDL_ttf library.

      @note This class is private; it cannot be directly instantiated,
         it must be created through @a TTF.OpenFont
   */

   Falcon::Symbol *c_ttffont = self->addClass( "TTFFont" );
   c_ttffont->setWKS( true );
   c_ttffont->exported( false ); // It's private.
   c_ttffont->getClassDef()->setObjectManager( &Falcon::core_falcon_data_manager );

   self->addClassMethod( c_ttffont, "GetFontStyle", Falcon::Ext::ttf_GetFontStyle );
   self->addClassMethod( c_ttffont, "SetFontStyle", Falcon::Ext::ttf_SetFontStyle ).asSymbol()->
      addParam("style");
   self->addClassMethod( c_ttffont, "FontHeight", Falcon::Ext::ttf_FontHeight );
   self->addClassMethod( c_ttffont, "FontAscent", Falcon::Ext::ttf_FontAscent );
   self->addClassMethod( c_ttffont, "FontDescent", Falcon::Ext::ttf_FontDescent );
   self->addClassMethod( c_ttffont, "FontLineSkip", Falcon::Ext::ttf_FontLineSkip );
   self->addClassMethod( c_ttffont, "FontFaces", Falcon::Ext::ttf_FontFaces );
   self->addClassMethod( c_ttffont, "FontFaceIsFixedWidth", Falcon::Ext::ttf_FontFaceIsFixedWidth );
   self->addClassMethod( c_ttffont, "FontFaceFamilyName", Falcon::Ext::ttf_FontFaceFamilyName );
   self->addClassMethod( c_ttffont, "FontFaceStyleName", Falcon::Ext::ttf_FontFaceStyleName );
   self->addClassMethod( c_ttffont, "GlyphMetrics", Falcon::Ext::ttf_GlyphMetrics ).asSymbol()->
      addParam("charId")->addParam("metrics");
   self->addClassMethod( c_ttffont, "SizeText", Falcon::Ext::ttf_SizeText ).asSymbol()->
      addParam("string")->addParam("metrics");
   self->addClassMethod( c_ttffont, "Render_Solid", Falcon::Ext::ttf_Render_Solid ).asSymbol()->
      addParam("string")->addParam("color");
   self->addClassMethod( c_ttffont, "Render_Shaded", Falcon::Ext::ttf_Render_Shaded ).asSymbol()->
      addParam("string")->addParam("color")->addParam("bgcolor");
   self->addClassMethod( c_ttffont, "Render_Blended", Falcon::Ext::ttf_Render_Blended ).asSymbol()->
      addParam("string")->addParam("color");

   /*#
      @class TTFMetrics
      @brief Metrics for glyphs and rendered strings.

      This class stores porperties that are used by font metrics
      related functions and metods to report sizes of font
      renderings.

      Those functions doesn't use directly this class; instead,
      they check the interface of the objects that they receive,
      so any class providing the required properties can be used.

      @prop minx minimum x value for a glyph
      @prop miny minimum y value for a glyph
      @prop maxx maximum x value for a glyph
      @prop maxy maximum y value for a glyph
      @prop ascent Asccent value for a glyph
      @prop w width of a glyph or string rendering
      @prop h height of a glyph or string rendering
   */

   Falcon::Symbol *c_ttfmetrics = self->addClass( "TTFMetrics" );
   c_ttffont->setWKS( true );
   self->addClassProperty( c_ttfmetrics, "minx" );
   self->addClassProperty( c_ttfmetrics, "miny" );
   self->addClassProperty( c_ttfmetrics, "maxx" );
   self->addClassProperty( c_ttfmetrics, "maxy" );
   self->addClassProperty( c_ttfmetrics, "ascent" );
   self->addClassProperty( c_ttfmetrics, "w" );
   self->addClassProperty( c_ttfmetrics, "h" );

   //=================================================================
   // Auto quit feature
   //
   Falcon::Symbol *c_sdl_aq = self->addClass( "_TTF_AutoQuit" );
   c_sdl_aq->setWKS( true );
   c_sdl_aq->exported( false );
   c_sdl_aq->getClassDef()->setObjectManager( &Falcon::core_falcon_data_manager );
   self->addClassMethod( c_sdl_aq, "Quit", Falcon::Ext::ttf_Quit );


   return self;
}

/* end of sdlttf.cpp */

