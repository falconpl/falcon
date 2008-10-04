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

}
}

#endif

/* end of sdlmixer_ext.h */
