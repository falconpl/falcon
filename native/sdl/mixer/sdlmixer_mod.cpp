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
#define FALCON_EXPORT_MODULE

#include <falcon/vm.h>
#include <falcon/vmmsg.h>
#include <falcon/membuf.h>
#include <sdl_service.h>
#include "sdlmixer_mod.h"

namespace Falcon {
namespace Ext {


//=========================================================
// Module-wide part
//
VMachine* m_channel_listener = 0;
VMachine* m_music_listener = 0;

Mutex* m_mtx_listener;


//=========================================================


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

void MixChunkCarrier::gcMark( uint32 )
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

void MixMusicCarrier::gcMark( uint32 )
{
   // noop
}

FalconData* MixMusicCarrier::clone() const
{
   return new MixMusicCarrier( *this );
}

}
}

void falcon_sdl_mixer_on_channel_done( int channel )
{
   // Very probably this increffing here is an overkill,
   // but it's nice to be on the bright side.

   Falcon::Ext::m_mtx_listener->lock();
   Falcon::VMachine* vm = Falcon::Ext::m_channel_listener;
   if( vm == 0 )
   {
       Falcon::Ext::m_mtx_listener->unlock();
       return;
   }
   vm->incref();
   Falcon::Ext::m_mtx_listener->unlock();

   Falcon::VMMessage *msg = new Falcon::VMMessage( "sdl_ChannelFinished" );
   msg->addParam( (Falcon::int64) channel );
   vm->postMessage( msg );
   vm->decref();
}

void falcon_sdl_mixer_on_music_finished()
{
   // Very probably this increffing here is an overkill,
   // but it's nice to be on the bright side.

   Falcon::Ext::m_mtx_listener->lock();
   Falcon::VMachine* vm = Falcon::Ext::m_music_listener;
   if( vm == 0 )
   {
      Falcon::Ext:: m_mtx_listener->unlock();
       return;
   }
   vm->incref();
   Falcon::Ext::m_mtx_listener->unlock();

   Falcon::VMMessage *msg = new Falcon::VMMessage( "sdl_MusicFinished" );
   vm->postMessage( msg );
   vm->decref();
}

/* end of sdlmixer_mod.cpp */
