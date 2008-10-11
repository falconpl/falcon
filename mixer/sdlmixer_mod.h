/*
   FALCON - The Falcon Programming Language.
   FILE: sdlmixer_mod.h

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

#ifndef FALCON_SDLMIXER_MOD
#define FALCON_SDLMIXER_MOD

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include <falcon/error.h>

extern "C" {
   #include <SDL_mixer.h>
}

#ifndef FALCON_SDLMIXER_ERROR_BASE
#define FALCON_SDLMIXER_ERROR_BASE 2140
#endif

namespace Falcon {
namespace Ext {

class MixChunkCarrier: public FalconData {
private:
   Mix_Chunk* m_chunk;
   int32 *m_counter;

public:
   MixChunkCarrier( Mix_Chunk* c );
   MixChunkCarrier( const MixChunkCarrier &other );

   virtual ~MixChunkCarrier();
   virtual void gcMark( VMachine* );
   virtual FalconData* clone() const;

   Mix_Chunk* chunk() const { return m_chunk; }
};

class MixMusicCarrier: public FalconData {
private:
   Mix_Music* m_music;
   int32 *m_counter;

public:
   MixMusicCarrier( Mix_Music* m );
   MixMusicCarrier( const MixMusicCarrier &other );

   virtual ~MixMusicCarrier();
   virtual void gcMark( VMachine* );
   virtual FalconData* clone() const;

   Mix_Music* chunk() const { return m_music; }
};

}
}

extern "C" void falcon_sdl_mixer_on_channel_done( int channel );
extern "C" void falcon_sdl_mixer_on_music_finished();

#endif

/* end of sdlttf_mod.h */
