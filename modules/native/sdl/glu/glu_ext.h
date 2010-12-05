/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_ext.h

   The SDL True Type binding support module.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Tue, 12 Aug 2009 00:06:56 +1100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL OpenGL binding support module.
*/

#ifndef flc_sdlopengl_ext_H
#define flc_sdlopengl_ext_H

#include <falcon/setup.h>


namespace Falcon {
namespace Ext {

//TODO: Put openGL functions here once bound
   FALCON_FUNC openglu_Perspective( VMachine *vm );
   FALCON_FUNC openglu_Build2DMipmaps( VMachine *vm );
   FALCON_FUNC openglu_ErrorString( VMachine *vm );

}
}

#endif

/* end of sdlopengl_ext.h */
