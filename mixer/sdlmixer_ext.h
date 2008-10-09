/*
   FALCON - The Falcon Programming Language.
   FILE: sdlmixer_ext.h

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

#ifndef flc_sdlmix_ext_H
#define flc_sdlmix_ext_H

#include <falcon/setup.h>

namespace Falcon {
namespace Ext {

// Generic mixer
FALCON_FUNC mix_Compiled_Version( VMachine *vm );
FALCON_FUNC mix_Linked_Version( VMachine *vm );
FALCON_FUNC mix_OpenAudio( VMachine *vm );
FALCON_FUNC mix_CloseAudio( VMachine *vm );
FALCON_FUNC mix_QuerySpec( VMachine *vm );

// waves
FALCON_FUNC mix_LoadWAV( VMachine *vm );

// channels
FALCON_FUNC mix_AllocateChannels( VMachine *vm );
FALCON_FUNC mix_Volume( VMachine *vm );

//==========================================
// Mix Chunks
FALCON_FUNC MixChunk_init( VMachine *vm );
FALCON_FUNC MixChunk_Volume( VMachine *vm );
FALCON_FUNC MixChunk_Play( VMachine *vm );

//==========================================
// Mix Music
FALCON_FUNC MixMusic_init( VMachine *vm );

}
}

#endif

/* end of sdlmixer_ext.h */
