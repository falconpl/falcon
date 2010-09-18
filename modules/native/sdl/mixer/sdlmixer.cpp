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
#include <falcon/module.h>
#include <falcon/vm.h>
#include "version.h"
#include "sdlmixer_ext.h"
#include "sdlmixer_mod.h"

namespace Falcon {
namespace Ext {

class SDLMixerModule: public Module
{
public:
   SDLMixerModule();
   virtual ~SDLMixerModule();
};

SDLMixerModule::SDLMixerModule()
{
   m_mtx_listener = new Mutex;
}

SDLMixerModule::~SDLMixerModule()
{
   ::Mix_HookMusicFinished( NULL );
   ::Mix_ChannelFinished( NULL );

   m_mtx_listener->lock();

   if ( m_channel_listener != 0 )
   {
      m_channel_listener->decref();
   }

   if ( m_music_listener != 0 )
   {
      m_music_listener->decref();
   }

   m_mtx_listener->unlock();

   delete m_mtx_listener;

}
}
}
/*#
   @module sdlmixer SDL Audio Mixer
   @brief SDL AUDIO extensions for the Falcon SDL module.
   
   This module encapsulates the sdl_mixer extensions for
   SDL library.
   
   @beginmodule sdlmixer
*/


FALCON_MODULE_DECL
{
   Falcon::Module *self = new Falcon::Ext::SDLMixerModule();
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
      @brief Enumeartion listing the possible audio open modes.

      Possible values:

      - U8: Unsigned 8-bit samples
      - S8: Signed 8-bit samples
      - U16LSB: Unsigned 16-bit samples, in little-endian byte order
      - S16LSB: Signed 16-bit samples, in little-endian byte order
      - U16MSB: Unsigned 16-bit samples, in big-endian byte order
      - S16MSB: Signed 16-bit samples, in big-endian byte order
      - U16: same as AUDIO_U16LSB (for backwards compatability probably)
      - S16: same as AUDIO_S16LSB (for backwards compatability probably)
      - U16SYS: Unsigned 16-bit samples, in system byte order
      - S16SYS: Signed 16-bit samples, in system byte order
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
      @enum MUS
      @brief Enumeartion listing the possible music types.

      Possible values:

      - NONE
      - CMD: external command
      - WAV: Pure wave
      - MOD: FastTracker MOD
      - MID: Midi
      - OGG: Ogg-vorbis
      - MP3: MP3
      - MP3_MAD: MP3 via MAD library.
   */
   Falcon::Symbol *c_mus_t= self->addClass( "MUS" );
   self->addClassProperty( c_mus_t, "NONE").setInteger( MUS_NONE );
   self->addClassProperty( c_mus_t, "CMD").setInteger( MUS_CMD );
   self->addClassProperty( c_mus_t, "WAV").setInteger( MUS_WAV );
   self->addClassProperty( c_mus_t, "MOD").setInteger( MUS_MOD );
   self->addClassProperty( c_mus_t, "MID").setInteger( MUS_MID );
   self->addClassProperty( c_mus_t, "OGG").setInteger( MUS_OGG );
   self->addClassProperty( c_mus_t, "MP3").setInteger( MUS_MP3 );
   self->addClassProperty( c_mus_t, "MP3_MAD").setInteger( MUS_MP3_MAD );

   /*#
      @class MIX
      @brief Main SDL Mixer encapsulation class.

      This class is the namespace for MIX functions of the SDL module.
      It contains the extensions provided by Falcon on the sdlmix
      module.
   */

   Falcon::Symbol *c_sdlmix = self->addClass( "MIX" );

   // Definitions
   self->addClassProperty( c_sdlmix, "DEFAULT_FREQUENCY" ).setInteger( MIX_DEFAULT_FREQUENCY );
   self->addClassProperty( c_sdlmix, "MAX_VOLUME" ).setInteger( MIX_MAX_VOLUME );
   self->addClassProperty( c_sdlmix, "NO_FADING" ).setInteger( MIX_NO_FADING );
   self->addClassProperty( c_sdlmix, "FADING_OUT" ).setInteger( MIX_FADING_OUT );
   self->addClassProperty( c_sdlmix, "FADING_IN" ).setInteger( MIX_FADING_IN );

   // Init and quit
   self->addClassMethod( c_sdlmix, "Compiled_Version", Falcon::Ext::mix_Compiled_Version );
   self->addClassMethod( c_sdlmix, "Linked_Version", Falcon::Ext::mix_Linked_Version );
   self->addClassMethod( c_sdlmix, "OpenAudio", Falcon::Ext::mix_OpenAudio ).asSymbol()->
      addParam("frequency")->addParam("format")->addParam("channels")->addParam("chunksize");
   self->addClassMethod( c_sdlmix, "CloseAudio", Falcon::Ext::mix_CloseAudio );
   self->addClassMethod( c_sdlmix, "QuerySpec", Falcon::Ext::mix_QuerySpec );

   // waves
   self->addClassMethod( c_sdlmix, "LoadWAV", Falcon::Ext::mix_LoadWAV ).asSymbol()->
      addParam("filename");

   // channels
   self->addClassMethod( c_sdlmix, "AllocateChannels", Falcon::Ext::mix_AllocateChannels ).asSymbol()->
      addParam("channels");
   self->addClassMethod( c_sdlmix, "Volume", Falcon::Ext::mix_Volume ).asSymbol()->
      addParam("channel")->addParam("volume");

   self->addClassMethod( c_sdlmix, "Pause", Falcon::Ext::mix_Pause ).asSymbol()->
      addParam("channel");
   self->addClassMethod( c_sdlmix, "Resume", Falcon::Ext::mix_Resume ).asSymbol()->
      addParam("channel");
   self->addClassMethod( c_sdlmix, "HaltChannel", Falcon::Ext::mix_HaltChannel ).asSymbol()->
      addParam("channel");
   self->addClassMethod( c_sdlmix, "ExpireChannel", Falcon::Ext::mix_ExpireChannel ).asSymbol()->
      addParam("channel")->addParam( "time" );
   self->addClassMethod( c_sdlmix, "FadeOutChannel", Falcon::Ext::mix_FadeOutChannel ).asSymbol()->
      addParam("channel")->addParam( "time" );
   self->addClassMethod( c_sdlmix, "ChannelFinished", Falcon::Ext::mix_ChannelFinished ).asSymbol()->
      addParam("active");

   self->addClassMethod( c_sdlmix, "Playing", Falcon::Ext::mix_Playing ).asSymbol()->
      addParam("channel");
   self->addClassMethod( c_sdlmix, "Paused", Falcon::Ext::mix_Paused ).asSymbol()->
      addParam("channel");
   self->addClassMethod( c_sdlmix, "FadingChannel", Falcon::Ext::mix_FadingChannel ).asSymbol()->
      addParam("channel");

   //===================================
   // Music
   self->addClassMethod( c_sdlmix, "LoadMUS", Falcon::Ext::mix_LoadMUS ).asSymbol()->
      addParam("file");
   self->addClassMethod( c_sdlmix, "VolumeMusic", Falcon::Ext::mix_VolumeMusic ).asSymbol()->
      addParam("volume");
   self->addClassMethod( c_sdlmix, "HaltMusic", Falcon::Ext::mix_HaltMusic );
   self->addClassMethod( c_sdlmix, "FadeOutMusic", Falcon::Ext::mix_FadeOutMusic ).asSymbol()->
      addParam("fadeOut");
   self->addClassMethod( c_sdlmix, "PauseMusic", Falcon::Ext::mix_PauseMusic );
   self->addClassMethod( c_sdlmix, "ResumeMusic", Falcon::Ext::mix_ResumeMusic );
   self->addClassMethod( c_sdlmix, "RewindMusic", Falcon::Ext::mix_RewindMusic );
   self->addClassMethod( c_sdlmix, "PausedMusic", Falcon::Ext::mix_PausedMusic );
   self->addClassMethod( c_sdlmix, "SetMusicPosition", Falcon::Ext::mix_SetMusicPosition ).asSymbol()->
      addParam("position");
   self->addClassMethod( c_sdlmix, "PlayingMusic", Falcon::Ext::mix_PlayingMusic );
   self->addClassMethod( c_sdlmix, "SetMusicCMD", Falcon::Ext::mix_SetMusicCMD ).asSymbol()->
      addParam("command");
   self->addClassMethod( c_sdlmix, "SetSynchroValue", Falcon::Ext::mix_SetSynchroValue ).asSymbol()->
      addParam("value");
   self->addClassMethod( c_sdlmix, "GetSynchroValue", Falcon::Ext::mix_GetSynchroValue );

   self->addClassMethod( c_sdlmix, "HookMusicFinished", Falcon::Ext::mix_HookMusicFinished ).asSymbol()->
      addParam("active");

   /*#
      @class MixChunk
      @brief SDL Mixer Chunk encapsulation class.

      This class is used to store chunks created by MIX_LoadWAV* functions.
      It contains chunk opaque data and it is mainly used as an input for
      MIX.Channel* methods.

      @see MIX.LoadWAV
   */

   Falcon::Symbol *c_sdlmix_chunk = self->addClass( "MixChunk", Falcon::Ext::MixChunk_init );
   c_sdlmix_chunk->setWKS( true );
   self->addClassMethod( c_sdlmix_chunk, "Volume", Falcon::Ext::MixChunk_Volume ).asSymbol()->
      addParam("volume");
   self->addClassMethod( c_sdlmix_chunk, "Play", Falcon::Ext::MixChunk_Play ).asSymbol()->
      addParam("channel")->addParam("loops")->addParam( "time" )->addParam( "fadeIn" );

   /*#
      @class MixMusic
      @brief SDL Mixer Music encapsulation class.

      This class is used to manipulate music (usually background music)
      in SDL Mixer. As SDL Mixer has its own channel for music, this class
      encapsulates all the functions related to SLD MIX_Music* functions
      (section 4.5 of the SDL Mixer API documentation).

      @see MIX.LoadMUS
   */
   Falcon::Symbol *c_sdlmix_music = self->addClass( "MixMusic", Falcon::Ext::MixMusic_init );
   c_sdlmix_music->setWKS( true );

   self->addClassMethod( c_sdlmix_music, "GetType", Falcon::Ext::MixMusic_GetType );
   self->addClassMethod( c_sdlmix_music, "Play", Falcon::Ext::MixMusic_Play ).asSymbol()->
      addParam("loops")->addParam( "fadeIn" )->addParam( "position" );

   return self;
}

/* end of sdlmixer.cpp */

