/*
   FALCON - The Falcon Programming Language.
   FILE: sdlmixer_mod.cpp

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

#include <falcon/vm.h>
#include <falcon/membuf.h>
#include <sdl_service.h>
#include "sdlmixer_mod.h"

namespace Falcon {
namespace Ext {

MixChunkCarrier::MixChunkCarrier( Mix_Chunk* c ):
   m_chunk( c )
{
   m_counter = (int32 *) memAlloc( sizeof( int32 ) );
}

MixChunkCarrier::MixChunkCarrier( const MixChunkCarrier &other )
{
   m_chunk = other.m_chunk;
   m_counter = other.m_counter;
   (*m_counter)++;
}

MixChunkCarrier::~MixChunkCarrier()
{
   (*m_counter)--;
   if( *m_counter <= 0 )
   {
      memFree( m_counter );
      Mix_FreeChunk( m_chunk );
   }
}

void MixChunkCarrier::gcMark( VMachine* )
{
   // noop
}

FalconData* MixChunkCarrier::clone() const
{
   return new MixChunkCarrier( *this );
}


//===========================================================
MixMusicCarrier::MixMusicCarrier( Mix_Music* c ):
   m_music( c )
{
   m_counter = (int32 *) memAlloc( sizeof( int32 ) );
}

MixMusicCarrier::MixMusicCarrier( const MixMusicCarrier &other )
{
   m_music = other.m_music;
   m_counter = other.m_counter;
   (*m_counter)++;
}

MixMusicCarrier::~MixMusicCarrier()
{
   (*m_counter)--;
   if( *m_counter <= 0 )
   {
      memFree( m_counter );
      Mix_FreeMusic( m_music );
   }
}

void MixMusicCarrier::gcMark( VMachine* )
{
   // noop
}

FalconData* MixMusicCarrier::clone() const
{
   return new MixMusicCarrier( *this );
}

}
}

#include "SDL.h"

void falcon_sdl_mixer_on_channel_done( int channel ) 
{
   // We must post a SDL user event for the main loop
   SDL_Event evt;
   evt.type = FALCON_SDL_CHANNEL_DONE_EVENT;
   evt.user.code = channel;
   ::SDL_PushEvent( &evt );
}

void falcon_sdl_mixer_on_music_finished()
{
   // We must post a SDL user event for the main loop
   SDL_Event evt;
   evt.type = FALCON_SDL_MUSIC_DONE_EVENT;
   ::SDL_PushEvent( &evt );
}

/* end of sdlmixer_mod.cpp */
