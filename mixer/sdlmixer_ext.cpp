/*
   FALCON - The Falcon Programming Language.
   FILE: sdlmixer_ext.cpp

   The SDL Mixer binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Oct 2008 19:11:27 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL Mixer binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "sdlmixer_ext.h"
#include "sdlmixer_mod.h"
#include <sdl_service.h>  // for the reset

extern "C" {
   #include <SDL_mixer.h>
}


/*# @beginmodule sdlmixer */

namespace Falcon {

static SDLService *s_service = 0;

namespace Ext {

/*#
   @method Compiled_Version MIX
//    @brief Determine the version used to compile this SDL mixer module.
   @return a three element array containing the major, minor and fix versions.
   @see MIX.Linked_Version
*/
FALCON_FUNC mix_Compiled_Version( VMachine *vm )
{
   SDL_version compile_version;
   MIX_VERSION(&compile_version);

   CoreArray *arr = new CoreArray( vm, 3 );
   arr->append( (int64) compile_version.major );
   arr->append( (int64) compile_version.minor );
   arr->append( (int64) compile_version.patch );
   vm->retval( arr );
}

/*#
   @method Linked_Version MIX
   @brief Determine the version of the library that is currently linked.
   @return a three element array containing the major, minor and fix versions.

   This function determines the version of the SDL_mixer library that is running
   on the system. As long as the interface is the same, it may be different
   from the version used to compile this module.
*/
FALCON_FUNC mix_Linked_Version( VMachine *vm )
{
   const SDL_version *link_version;
   link_version = Mix_Linked_Version();

   CoreArray *arr = new CoreArray( vm, 3 );
   arr->append( (int64) link_version->major );
   arr->append( (int64) link_version->minor );
   arr->append( (int64) link_version->patch );
   vm->retval( arr );
}


/*#
   @method OpenAudio MIX
   @brief Initialize the MIX module.
   @param frequency Output sampling frequency in samples per second (Hz).
          you might use MIX.DEFAULT_FREQUENCY(22050) since that is a good value for most games.
   @param format Output sample format; it's one of the @a MIX.AUDIO enums.
   @param channels Number of sound channels in output. Set to 2 for stereo, 1 for mono.
      This has nothing to do with mixing channels.
   @param chunksize Bytes used per output sample.
   @raise SDLError on initialization failure.

   It is necessary to call @a SDL.Init with SDL.INIT_AUDIO option.
*/

FALCON_FUNC mix_OpenAudio( VMachine *vm )
{
   Item *i_frequency = vm->param(0);
   Item *i_format = vm->param(1);
   Item *i_channels = vm->param(2);
   Item *i_chunksize = vm->param(3);

   if ( i_frequency == 0 || ! i_frequency->isOrdinal() ||
        i_format == 0 || ! i_format->isOrdinal() ||
        i_channels == 0 || ! i_channels->isOrdinal() ||
        i_chunksize == 0 || ! i_chunksize->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,N,N" ) ) );
      return;
   }

   int retval = ::Mix_OpenAudio(
      (int) i_frequency->forceInteger(),
      (uint16)i_format->forceInteger(),
      (int) i_channels->forceInteger(),
      (int) i_chunksize->forceInteger() );

   if ( retval != 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE, __LINE__ )
         .desc( "Mixer open" )
         .extra( Mix_GetError() ) ) );
      return;
   }

   // since we're here, let's load our service
   s_service = static_cast<SDLService*>(vm->getService( SDL_SERVICE_SIGNATURE ));
   fassert( s_service != 0 ); // or load should have failed.

   if ( s_service == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+1, __LINE__ )
         .desc( "SDL service not in the target VM" ) ) );
   }
}

/*#
   @method CloseAudio MIX
   @brief Turn off the MIX module.

   This method turns off he mixing facility. It must becalled
   the same number of times @a MIX.OpenAudio has been called.

   After the mix system has been effectively turned off, it is
   possible to re-init it.

   Also, the MIX module doesn't require MIX.CloseAudio() to be
   called before the SDL system is exited.
*/

FALCON_FUNC mix_CloseAudio( VMachine *vm )
{
  ::Mix_CloseAudio();
}

/*#
   @method QuerySpec MIX
   @brief Queries the settings that have been used to open the SDL mixer system.
   @return A vector with three numeric values representing the
      frequency, the audio format and the channel numbers that were set on init.
   @raise SDLError if the system was not initialized.

   @see MIX.OpenAudio
*/
FALCON_FUNC mix_QuerySpec( VMachine *vm )
{
   int res;
   int frequency;
   Uint16 format;
   int channels;

   res = ::Mix_QuerySpec(&frequency, &format, &channels);
   if ( res == 0 )
   {
      // not initialized.
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+1, __LINE__ )
         .desc( "Mixer not initialized" )
         .extra( Mix_GetError() ) ) );
      return;
   }

   CoreArray *retval = new CoreArray( vm, 3 );
   retval->append( (int64) frequency );
   retval->append( (int64) format );
   retval->append( (int64) channels );

   vm->retval( retval );
}

/*#
   @method LoadWAV MIX
   @brief Loads an audio file.
   @param file A file name or a stream to be loaded.
   @return On success, an instance of @a MixChunk class.
   @raise SDLError if the system was not initialized or on load error.

   The @b file parameter may be either a stream pointing to the beginning
   of a valid SDL_mixer supported file, or it may be a filename.

   If it's a string, the @b file parameter is not parsed through the
   Falcon I/O system; it's directly sent to the underlying SDL function,
   so a valid locally available file specification must be provided.

   @note This may change in future (we may use Falcon metafile services
   also to resolve file names).
   @see MIX.OpenAudio
*/
FALCON_FUNC mix_LoadWAV( VMachine *vm )
{
   Item *i_filename = vm->param(0);

   if ( i_filename == 0 ||
         ( ! i_filename->isString() &&
            !( i_filename->isObject() && i_filename->asObject()->derivedFrom("Stream") ))
      )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S|Stream" ) ) );
      return;
   }

   Mix_Chunk* chunk = 0;

   if( i_filename->isString() )
   {
      AutoCString filename( *i_filename->asString() );
      chunk = ::Mix_LoadWAV( filename.c_str() );
   }
   else {
      struct SDL_RWops rwops;
      Stream* stream = static_cast<Stream *>(i_filename->asObject()->getUserData());
      s_service->rwopsFromStream( rwops, stream );
      chunk = ::Mix_LoadWAV_RW( &rwops, 0 );
   }

   if ( chunk == 0 )
   {
      // not initialized.
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+2, __LINE__ )
         .desc( "Error in I/O operation" )
         .extra( Mix_GetError() ) ) );
      return;
   }

   Item *i_chunk_cls = vm->findWKI( "MixChunk" );
   fassert( i_chunk_cls != 0 && i_chunk_cls->isClass() );
   CoreObject* obj = i_chunk_cls->asClass()->createInstance();
   obj->setUserData( new MixChunkCarrier( chunk ) );

   vm->retval( obj );
}


//=======================================================================
// Channel control
//

/*#
   @method AllocateChannels MIX
   @brief Sets or read the number of mixing channels.
   @optparam channels Number the number of channels to be set, or nil to get current channel count.
   @return Current channel count (before eventually changing it).

   This method sets up mixing sound effect channels. Channels
   can be independently pre-processed and mixed at different volumes on the
   background music.

   This method can be called multiple times, even with sounds playing.
   If numchans is less than the current number of channels,
   then the higher channels will be stopped, freed, and therefore
   not mixed any longer. It's probably not a good idea to change
   the size 1000 times a second though.

   @note Passing in zero WILL free all mixing channels, however music will still play.

   Channels are also the place where MixChunk instances can be played.
*/
FALCON_FUNC mix_AllocateChannels( VMachine *vm )
{
   Item *i_channels = vm->param(0);

   if ( i_channels == 0 || i_channels->isNil() )
   {
      vm->retval( (int64) Mix_AllocateChannels(-1) );
   }
   else if( i_channels->isOrdinal() )
   {
      vm->retval( (int64) Mix_AllocateChannels((int) i_channels->forceInteger() ) );
   }
   else {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
   }
}


/*#
   @method Volume MIX
   @brief Sets or read current volume setting for given channel.
   @param channel Target channel ID
   @optparam volume The new volume to be set (pass nothing, nil or < 0 to read).
   @return Current volume level (prior to setting).

   This method convigure the overall volume setting
   mixing sound effect channels.

   Pass -1 as @b channel to set the volume in all the channels.
*/
FALCON_FUNC mix_Volume( VMachine *vm )
{
   Item *i_channel = vm->param(0);
   Item *i_volume = vm->param(1);

   if ( i_channel == 0 || ! i_channel->isOrdinal() ||
        (i_volume != 0 && ! i_volume->isNil() && ! i_volume->isOrdinal() ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,[N]" ) ) );
   }

   int channel = (int) i_channel->forceInteger();
   int volume = i_volume == 0 ? -1 :
      i_volume->isNil() ? -1 : ((int) i_volume->forceInteger() );

   vm->retval( (int64) Mix_Volume( channel, volume ) );
}

/*#
   @method Pause MIX
   @brief Pauses a channel.
   @param channel The channel to be paused (-1 for all).

   You can pause an already paused channel, and no error
   is raised for pausing an unexisting channel.
*/
FALCON_FUNC mix_Pause( VMachine *vm )
{
   Item *i_channel = vm->param(0);

   if ( i_channel == 0 || ! i_channel->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   ::Mix_Pause( (int) i_channel->forceInteger() );
}

/*#
   @method Resume MIX
   @brief Resumes a paused a channel.
   @param channel The channel to be resumed (-1 for all).

   You can resume an already playing channel, and no error
   is raised for resuming an unexisting channel.
*/
FALCON_FUNC mix_Resume( VMachine *vm )
{
   Item *i_channel = vm->param(0);

   if ( i_channel == 0 || ! i_channel->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   ::Mix_Resume( (int) i_channel->forceInteger() );
}


/*#
   @method HaltChannel MIX
   @brief Stops the playback on channel.
   @param channel The channel to be stopped (-1 for all).

   You can resume an already playing channel, and no error
   is raised for resuming an unexisting channel.
*/
FALCON_FUNC mix_HaltChannel( VMachine *vm )
{
   Item *i_channel = vm->param(0);

   if ( i_channel == 0 || ! i_channel->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   ::Mix_HaltChannel( (int) i_channel->forceInteger() );
}


/*#
   @method ExpireChannel MIX
   @brief Requests to the playback on a channel after some time.
   @param channel The channel to be stopped (-1 for all).
   @param time Number of seconds and fractions after which the timeout expires.
   @return Number of channels scheduled for stopping.

   @note The @b time parameter is in Falcon sleep format (i.e. 1.2 is 1.2 seconds).
*/
FALCON_FUNC mix_ExpireChannel( VMachine *vm )
{
   Item *i_channel = vm->param(0);
   Item *i_time = vm->param(1);

   if ( i_channel == 0 || ! i_channel->isOrdinal() ||
        i_time == 0 || ! i_time->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N" ) ) );
      return;
   }

   vm->retval( (int64) ::Mix_ExpireChannel( (int) i_channel->forceInteger(),
      (int) ( i_time->forceNumeric() * 1000.0 ) ) );
}

/*#
   @method FadeOutChannel MIX
   @brief Requests to the playback on a channel after some, fading out in the meanwhile.
   @param channel The channel to be faded (-1 for all).
   @param time Number of seconds and fractions after which the timeout expires.
   @return Number of channel fading out.

   @note The @b time parameter is in Falcon sleep format (i.e. 1.2 is 1.2 seconds).

*/
FALCON_FUNC mix_FadeOutChannel( VMachine *vm )
{
   Item *i_channel = vm->param(0);
   Item *i_time = vm->param(1);

   if ( i_channel == 0 || ! i_channel->isOrdinal() ||
        i_time == 0 || ! i_time->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N" ) ) );
      return;
   }

   vm->retval( (int64)  ::Mix_FadeOutChannel( (int) i_channel->forceInteger(),
      (int) ( i_time->forceNumeric() * 1000.0 ) ) );
}

/*#
   @method ChannelFinished MIX
   @brief Requests onChannelFinished callbacks to be called when a channel finishes playing.
   @param active Set to true to generate messages, false to prevent it.

   This method works differently with respect to SDL_Mixer C API counterpart.
 If activated, all channels terminating will generate a callback request on the
 SDL message queue. This will cause a waiting Falcon SDLEventHandler class to receive
 an onChannelFinished( chnum ) callback (where chnum is the number of the channel that
 has finished playing).
*/
FALCON_FUNC mix_ChannelFinished( VMachine *vm )
{
   Item *i_active = vm->param(0);
   if ( i_active == 0 )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "X" ) ) );
      return;
   }

   ::Mix_ChannelFinished( i_active->isTrue() ? falcon_sdl_mixer_on_channel_done : NULL );
}

/*#
   @method Playing MIX
   @brief Returns the number of playing channels.
   @optparam channel Channel ID that is to be queried (-1 for all).
   @return 1 if the desired channel is playing, or the number of playing
           channels channel is not passed or -1.

 Channel to test whether it is playing or not.
 Passing -1 in the @b channel parameter, or not passing it at all,
 will tell you how many channels are playing.
 Note: Does not check if the channel has been paused.
*/
FALCON_FUNC mix_Playing( VMachine *vm )
{
   Item *i_channel = vm->param(0);
   if ( i_channel != 0 &&
       ( ! i_channel->isOrdinal() || i_channel->isNil()) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   int channel = i_channel == 0 || i_channel->isNil() ? -1 :
      (int) i_channel->forceInteger();

   vm->retval( (int64) ::Mix_Playing( channel ) );
}


/*#
   @method Paused MIX
   @brief Returns the number of paused channels.
   @optparam channel Channel ID that is to be queried (-1 for all).
   @return 1 if the desired channel is paused, or the number of paused
           channels channel is not passed or -1.

 Channel to test whether it is paused or not.
 Passing -1 in the @b channel parameter, or not passing it at all,
 will tell you how many channels are paused.
*/
FALCON_FUNC mix_Paused( VMachine *vm )
{
   Item *i_channel = vm->param(0);
   if ( i_channel != 0 &&
       ( ! i_channel->isOrdinal() || i_channel->isNil()) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   int channel = i_channel == 0 || i_channel->isNil() ? -1 :
      (int) i_channel->forceInteger();

   vm->retval( (int64) ::Mix_Paused( channel ) );
}

/*#
   @method Paused MIX
   @brief Returns the number of channels currently fading out.
   @param channel Channel ID that is to be queried (-1 for all).
   @return One of the MIX.FADING_IN, MIX.FADING_OUT or MIX.NO_FADING values

   Channel to test whether it is faiding out or not.
*/

FALCON_FUNC mix_FadingChannel( VMachine *vm )
{
   Item *i_channel = vm->param(0);
   if ( i_channel == 0 || i_channel->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   int channel = (int) i_channel->forceInteger();

   if ( channel < 0 )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_param_range, __LINE__ ).
         extra( "N>0" ) ) );
      return;
   }

   vm->retval( (int64) ::Mix_FadingChannel( channel ) );
}

//=================================================================================
// Music support
//

/*#
   @method LoadMUS MIX
   @brief Loads a music file.
   @param file A file name or a stream to be loaded.
   @return On success, an instance of @a MixMusic class.
   @raise SDLError if the system was not initialized or on load error.

   The @b file parameter may be either a stream pointing to the beginning
   of a valid SDL_mixer supported file, or it may be a filename.

   If it's a string, the @b file parameter is not parsed through the
   Falcon I/O system; it's directly sent to the underlying SDL function,
   so a valid locally available file specification must be provided.

   @note This may change in future (we may use Falcon metafile services
   also to resolve file names).
   @see MIX.OpenAudio
*/
FALCON_FUNC mix_LoadMUS( VMachine *vm )
{
   Item *i_filename = vm->param(0);

   if ( i_filename == 0 ||
         ( ! i_filename->isString() &&
            !( i_filename->isObject() && i_filename->asObject()->derivedFrom("Stream") ))
      )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S|Stream" ) ) );
      return;
   }

   Mix_Music* music = 0;

   if( i_filename->isString() )
   {
      AutoCString filename( *i_filename->asString() );
      music = ::Mix_LoadMUS( filename.c_str() );
   }
   else {
      struct SDL_RWops rwops;
      Stream* stream = static_cast<Stream *>(i_filename->asObject()->getUserData());
      s_service->rwopsFromStream( rwops, stream );
      music = ::Mix_LoadMUS_RW( &rwops );
   }

   if ( music == 0 )
   {
      // not initialized.
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+2, __LINE__ )
         .desc( "Error in I/O operation" )
         .extra( Mix_GetError() ) ) );
      return;
   }

   Item *i_music_cls = vm->findWKI( "MixMusic" );
   fassert( i_music_cls != 0 && i_music_cls->isClass() );
   CoreObject* obj = i_music_cls->asClass()->createInstance();
   obj->setUserData( new MixMusicCarrier( music ) );

   vm->retval( obj );
}


/*#
   @method VolumeMusic MIX
   @brief Sets or read the global music volume.
   @optparam volume The new volume setting.
   @return Previous volume value.

   The volume setting goes from 0 to 127. If the @b volume parameter
   is not given, nil or -1, the setting is just read without being changed.
*/
FALCON_FUNC mix_VolumeMusic( VMachine *vm )
{
   Item *i_volume = vm->param(0);

   if ( i_volume != 0 && ! i_volume->isOrdinal() && ! i_volume->isNil() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   int volume = i_volume == 0 || i_volume->isNil() ? -1 : i_volume->forceInteger();

   vm->retval( (int64) ::Mix_VolumeMusic( volume ) );
}

/*#
   @method HaltMusic MIX
   @brief Immediately halts the music.

   (If it's playing...)
*/
FALCON_FUNC mix_HaltMusic( VMachine *vm )
{
   ::Mix_HaltMusic();
}


/*#
   @method FadeOutMusic MIX
   @brief Fades out the music channel.
   @param fadeOut Duration of the fadeout time in seconds and fractions.
*/
FALCON_FUNC mix_FadeOutMusic( VMachine *vm )
{
   Item *i_fadeout = vm->param(0);

   if ( i_fadeout == 0 || ! i_fadeout->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   ::Mix_FadeOutMusic( (int) (i_fadeout->forceNumeric()*1000.0) );
}


/*#
   @method PauseMusic MIX
   @brief Pause the music.

   The music can later be resumed from the current position though
   @a MIX.ResumeMusic.

   @see MIX.PausedMusic
*/
FALCON_FUNC mix_PauseMusic( VMachine *vm )
{
   ::Mix_PauseMusic();
}

/*#
   @method ResumeMusic MIX
   @brief Resumes a paused music.

   @see MIX.PauseMusic
*/
FALCON_FUNC mix_ResumeMusic( VMachine *vm )
{
   ::Mix_ResumeMusic();
}

/*#
   @method RewindMusic MIX
   @brief Reposition music stream at the start of the currently played music.

   @see MIX.SetMusicPosition
*/
FALCON_FUNC mix_RewindMusic( VMachine *vm )
{
   ::Mix_RewindMusic();
}

/*#
   @method RewindMusic MIX
   @brief Determines if the music is currently playing but paused.
   @return True if the music is paused.

   @see MIX.PauseMusic
*/
FALCON_FUNC mix_PausedMusic( VMachine *vm )
{
   vm->regA().setBoolean( ::Mix_PausedMusic() != 0 );
}

/*#
   @method SetMusicPosition MIX
   @brief Changes the position in the music stream.
   @param position The relative position (0 to 1).
   @raise SDLError if the function is not implemented
      for the current music stream type.
*/
FALCON_FUNC mix_SetMusicPosition( VMachine *vm )
{
   Item* i_position = vm->param(0);

   if ( i_position == 0 || ! i_position->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   if ( ::Mix_SetMusicPosition( i_position->forceNumeric() ) == 0 )
   {
      vm->raiseModError( new  SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+5, __LINE__ ).
         desc( "Not implemented").
         extra( "SetMusicPosition" ) ) );
   }
}


/*#
   @method PlayingMusic MIX
   @brief Checks if the music is playing or not.
   @return True if it's playing, false otherwhise.

   Returns true also if the music is paused.
*/

FALCON_FUNC mix_PlayingMusic( VMachine *vm )
{
   vm->regA().setBoolean( ::Mix_PlayingMusic() != 0 );
}

/*#
   @method SetMusicCMD MIX
   @brief Sets the command used to load some kind of music.
   @raise SDLError In case of problems in finding the given command.

   See the SDL_mixer api description.
*/
FALCON_FUNC mix_SetMusicCMD( VMachine *vm )
{
   Item* i_command = vm->param(0);

   if ( i_command == 0 || ! i_command->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   AutoCString command( *i_command->asString() );
   if( ::Mix_SetMusicCMD( command.c_str() ) == 0 )
   {
      vm->raiseModError( new  SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+2, __LINE__ )
         .desc( "Error in I/O operation" )
         .extra( Mix_GetError() ) ) );
   }
}

/*#
   @method SetSynchroValue MIX
   @brief Changes the Synchro value for MICMOD library (Mod, STM, IT, etc.).
   @param value New synchro value.

   See the SDL_mixer API description.
*/
FALCON_FUNC mix_SetSynchroValue( VMachine *vm )
{
   Item* i_value = vm->param(0);

   if ( i_value == 0 || ! i_value->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   ::Mix_SetSynchroValue( (int) i_value->forceInteger() );
}


/*#
   @method GetSynchroValue MIX
   @brief Returns Syncro value for MICMOD library (Mod, STM, IT, etc.).
   @return Current Synchro value.

   See the SDL_mixer API description.
*/
FALCON_FUNC mix_GetSynchroValue( VMachine *vm )
{
   vm->regA().setInteger( ::Mix_GetSynchroValue() );
}


/*#
   @method HookMusicFinished MIX
   @brief Requests onMusicFinished callbacks to be called when the music finishes playing.
   @param active Set to true to generate messages, false to prevent it.

   This method works differently with respect to SDL_Mixer C API counterpart.
 If activated, the music terminating or being terminated will generate a callback request on the
 SDL message queue. This will cause a waiting Falcon SDLEventHandler class to receive
 an onMusicFinished() callback.
*/
FALCON_FUNC mix_HookMusicFinished( VMachine *vm )
{
   Item *i_active = vm->param(0);
   if ( i_active == 0  )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "X" ) ) );
      return;
   }

   if( i_active->isTrue() )
      ::Mix_HookMusicFinished( falcon_sdl_mixer_on_music_finished );
   else
      ::Mix_HookMusicFinished( NULL );
}

//=======================================================================
// Mix chunks
//
FALCON_FUNC MixChunk_init( VMachine *vm )
{
   vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+3, __LINE__ )
      .desc( "Can't instantiate directly this class" ) ) );
}

/*#
   @method Volume MixChunk
   @brief Sets the mixing volume for a loaded chunk.
   @optparam volume Volume level between 0 and MIX.MAX_VOLUME (128).
   @return Previous setting for volume.

   If @b volume is set to a less than zero integer, or if its not given,
   the previous value for this setting is returned and the value is not
   changed.
*/
FALCON_FUNC MixChunk_Volume( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Mix_Chunk* chunk = static_cast<MixChunkCarrier *>(self->getUserData())->chunk();

   Item *i_volume = vm->param(0);
   if( i_volume == 0 || i_volume->isNil() )
   {
      vm->retval( (int64) Mix_VolumeChunk( chunk, -1 ) );
   }
   else if ( ! i_volume->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
   }
   else {
      vm->retval( (int64) Mix_VolumeChunk( chunk, (int) i_volume->forceInteger() ) );
   }
}

/*#
   @method Play MixChunk
   @brief Play a sound on a given channel
   @param channel Target channel ID (-1 to select the first available channel).
   @optparam loops Numbers of repetitions; 1 means repeat once, -1 repeat forever.
   @optparam time Seconds and fractions during which music will play.
   @optparam fadeIn Seconds and fractions for the fade-in effect.
   @return The channel on which the sound is played.
   @raise SDLError on playback error.

   This method plays a previously loaded sound onto one channel.

   If the @b loops parameter is not given, the chunk will play just once.

   If @b fadeIn parameter is not given, nil or <=0, the sample will play immediately
   at full volume without fade-in.

   If @b time parameter is not given, nil or -1, the sample will play forever,
   until the channel is stopped, while if it's a value, it will play for the
   given amount of seconds.

   @note This method encapsulates the functions of Mix_PlayChannel, Mix_PlayChannelTimed,
   Mix_FadeInChannel and Mix_FadeInChannelTimed in the SDL_Mixere API.
*/
FALCON_FUNC MixChunk_Play( VMachine *vm )
{
   Item *i_channel = vm->param(0);
   Item *i_loops = vm->param(1);
   Item *i_time = vm->param(2);
   Item *i_fadeIn = vm->param(3);

   if ( i_channel == 0 || ! i_channel->isOrdinal() ||
        (i_loops != 0 && ! i_loops->isNil() && ! i_loops->isOrdinal()) ||
        (i_time != 0 && ! i_time->isNil() && ! i_time->isOrdinal()) ||
        (i_fadeIn != 0 && ! i_fadeIn->isNil() && ! i_fadeIn->isOrdinal()) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,[N],[N],[N]" ) ) );
   }

   CoreObject *self = vm->self().asObject();
   Mix_Chunk* chunk = static_cast<MixChunkCarrier *>(self->getUserData())->chunk();

   int channel = (int) i_channel->forceInteger();
   int loops = i_loops == 0 || i_loops->isNil() ? 1: (int) i_loops->forceInteger();
   int res;

   if ( i_fadeIn == 0 || i_fadeIn->isNil() )
   {
      res = i_time == 0 || i_time->isNil() ?
         Mix_PlayChannel( channel, chunk, loops ) :
         Mix_PlayChannelTimed( channel, chunk, loops, (int)(i_time->forceNumeric() * 1000.0 ));
   }
   else {
      int ms = (int)(i_fadeIn->forceNumeric() * 1000.0);
      res = i_time == 0 || i_time->isNil() ?
         Mix_FadeInChannel( channel, chunk, loops, ms ) :
         Mix_FadeInChannelTimed( channel, chunk, loops, ms, (int)(i_time->forceNumeric() * 1000.0 ));
   }

   if ( res < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+4, __LINE__ )
         .desc( "Playback error" )
         .extra( Mix_GetError() ) ) );
      return;
   }

   vm->retval( (int64) res );
}

//=======================================================================
// Mix music
//
FALCON_FUNC MixMusic_init( VMachine *vm )
{
   vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+3, __LINE__ )
      .desc( "Can't instantiate directly this class" ) ) );
}

/*#
   @method Play GetType
   @brief Return the loaded music type.
   @return One of the @a MUS enum items.
*/
FALCON_FUNC MixMusic_GetType( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Mix_Music* music = static_cast<MixMusicCarrier *>(self->getUserData())->music();
   vm->retval( (int64) Mix_GetMusicType( music ) );
}

/*#
   @method Play MixMusic
   @brief Play a sound on a given channel
   @optparam loops Numbers of repetitions; 1 means repeat once, -1 repeat forever.
   @optparam fadeIn Seconds and fractions for the fade-in effect.
   @optparam position Relative position in the music stream (0 to 1).
   @raise SDLError on playback error.

   This method plays a previously loaded music information.

   If the @b loops parameter is not given, the music will play just once.

   If @b fadeIn parameter is not given, nil or <=0, the music will play immediately
   at full volume without fade-in.

   @note This method encapsulates the functions of Mix_PlayMusic, Mix_FadeInMusic and
   Mix_cFadeInMusiPos in the SDL_Mixere API.
*/
FALCON_FUNC MixMusic_Play( VMachine *vm )
{
   Item *i_loops = vm->param(0);
   Item *i_fadeIn = vm->param(1);
   Item *i_position = vm->param(2);

   if ( (i_loops != 0 && ! i_loops->isNil() && ! i_loops->isOrdinal()) ||
        (i_fadeIn != 0 && ! i_fadeIn->isNil() && ! i_fadeIn->isOrdinal()) ||
        (i_position != 0 && ! i_position->isNil() && ! i_position->isOrdinal())
      )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N],[N],[N]" ) ) );
   }

   CoreObject *self = vm->self().asObject();
   Mix_Music* music = static_cast<MixMusicCarrier *>(self->getUserData())->music();

   int loops = i_loops == 0 || i_loops->isNil() ? 1: (int) i_loops->forceInteger();
   int res;

   if ( i_fadeIn == 0 || i_fadeIn->isNil() )
   {
      res = ::Mix_PlayMusic( music, loops );
   }
   else {
      int ms = (int)(i_fadeIn->forceNumeric() * 1000.0);
      res = i_position == 0 || i_position->isNil() ?
         ::Mix_FadeInMusic( music, loops, ms ) :
         ::Mix_FadeInMusicPos( music, loops, ms, (int)(i_position->forceNumeric() * 1000.0 ));
   }

   if ( res < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDLMIXER_ERROR_BASE+4, __LINE__ )
         .desc( "Playback error" )
         .extra( Mix_GetError() ) ) );
      return;
   }
}


}
}

/* end of MIX_ext.cpp */
