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

   // Init and quit
   /*#
      @enum AUDIO
      @brief Enumeartion listing the possible numeric error codes raised by MXML.

      This enumeration contains error codes which are set as values for the
      code field of the MXMLError raised in case of processing or I/O error.

      - @b Io: the operation couldn't be completed because of a physical error
         on the underlying stream.
      - @b Nomem: MXML couldn't allocate enough memory to complete the operation.
      - @b OutChar: Invalid characters found between tags.
      - @b InvalidNode: The node name contains invalid characters.
      - @b InvalidAtt: The attribute name contains invalid characters.
      - @b MalformedAtt: The attribute declaration doesn't conform XML standard.
      - @b InvalidChar: Character invalid in a certain context.
      - @b Unclosed: A node was open but not closed.
      - @b UnclosedEntity: The '&' entity escape was not balanced by a ';' closing it.
      - @b WrongEntity: Invalid entity name.
      - @b AttrNotFound: Searched attribute was not found.
      - @b ChildNotFound: Searched child node was not found.
      - @b Hyerarcy: Broken hierarcy; given node is not in a valid tree.
      - @b CommentInvalid: The comment node is not correctly closed by a --> sequence.
      - @b MultipleXmlDecl: the PI ?xml is declared more than once, or after another node.
   */
   Falcon::Symbol *c_audio = self->addClass( "AUDIO" );
   self->addClassProperty( c_audio, "U8").setInteger( AUDIO_U8 );
   self->addClassProperty( c_audio, "S8").setInteger( AUDIO_S8 );
   self->addClassProperty( c_audio, "U16LSB").setInteger( AUDIO_U16LSB );
   self->addClassProperty( c_audio, "S16LSB").setInteger( AUDIO_S16LSB );
   self->addClassProperty( c_audio, "U16MSB").setInteger( AUDIO_U16MSB );
   self->addClassProperty( c_audio, "S16MSB").setInteger( AUDIO_S16MSB );
   self->addClassProperty( c_audio, "U16").setInteger( AUDIO_U16 );
   self->addClassProperty( c_audio, "S16").setInteger( AUDIO_S16 );
   self->addClassProperty( c_audio, "U16SYS").setInteger( AUDIO_U16SYS );
   self->addClassProperty( c_audio, "S16SYS").setInteger( AUDIO_S16SYS );

   /*#
      @class MIX
      @brief Main SDL Mixer encapsulation class.

      This class is the namespace for MIX functions of the SDL module.
      It contains the extensions provided by Falcon on the sdlmix
      module.
   */

   Falcon::Symbol *c_sdlmix = self->addClass( "MIX" );
   // Init and quit
   self->addClassProperty( c_sdlmix, "DEFAULT_FREQUENCY" ).setInteger( MIX_DEFAULT_FREQUENCY );
   self->addClassMethod( c_sdlmix, "Compiled_Version", Falcon::Ext::mix_Compiled_Version );
   self->addClassMethod( c_sdlmix, "Linked_Version", Falcon::Ext::mix_Linked_Version );
   self->addClassMethod( c_sdlmix, "OpenAudio", Falcon::Ext::mix_OpenAudio ).asSymbol()->
      addParam("frequency")->addParam("format")->addParam("channels")->addParam("chunksize");
   self->addClassMethod( c_sdlmix, "CloseAudio", Falcon::Ext::mix_CloseAudio );

   return self;
}

/* end of sdlmixer.cpp */

