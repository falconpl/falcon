/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_ext.h

   The SDL True Type binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:11:06 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   The SDL True Type binding support module.
*/

#ifndef flc_sdlttf_ext_H
#define flc_sdlttf_ext_H

#include <falcon/setup.h>


namespace Falcon {
namespace Ext {

FALCON_FUNC ttf_Init( VMachine *vm );
FALCON_FUNC ttf_WasInit( VMachine *vm );
FALCON_FUNC ttf_InitAuto( VMachine *vm );
FALCON_FUNC ttf_Quit( VMachine *vm );
FALCON_FUNC ttf_Compiled_Version( VMachine *vm );
FALCON_FUNC ttf_Linked_Version( VMachine *vm );


}
}

#endif

/* end of sdlttf_ext.h */
