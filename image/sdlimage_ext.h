/*
   FALCON - The Falcon Programming Language.
   FILE: sdlimage_ext.h

   The SDL image loading binding support module.
   -------------------------------------------------------------------
   Author: Federico Baroni
   Begin: Tue, 30 Sep 2008 23:40:06 +0100

   Last modified because:
   Tue 7 Oct 2008 23:06:03 - GetError and SetError added

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL image loading binding support module.
*/

#ifndef flc_sdlimage_ext_H
#define flc_sdlimage_ext_H

#include <falcon/setup.h>


namespace Falcon {
namespace Ext {

// Loading functions

FALCON_FUNC img_Load ( VMachine *vm );
//FALCON_FUNC img_Load_RW ( VMachine *vm );
//FALCON_FUNC img_LoadTyped_RW ( VMachine *vm );
//FALCON_FUNC img_LoadBMP_RW ( VMachine *vm );
//FALCON_FUNC img_LoadPNM_RW ( VMachine *vm );
//FALCON_FUNC img_LoadXPM_RW ( VMachine *vm );
//FALCON_FUNC img_LoadXCF_RW ( VMachine *vm );
//FALCON_FUNC img_LoadPCX_RW ( VMachine *vm );
//FALCON_FUNC img_LoadGIF_RW ( VMachine *vm );
//FALCON_FUNC img_LoadJPG_RW ( VMachine *vm );
//FALCON_FUNC img_LoadTIF_RW ( VMachine *vm );
//FALCON_FUNC img_LoadPNG_RW ( VMachine *vm );
//FALCON_FUNC img_LoadTGA_RW ( VMachine *vm );
//FALCON_FUNC img_LoadLBM_RW ( VMachine *vm );
//FALCON_FUNC img_ReadXPMFromArray ( VMachine *vm );

// Info functions

//FALCON_FUNC img_isBMP ( VMachine *vm );
//FALCON_FUNC img_isPNM ( VMachine *vm );
//FALCON_FUNC img_isXPM ( VMachine *vm );
//FALCON_FUNC img_isXCF ( VMachine *vm );
//FALCON_FUNC img_isPCX ( VMachine *vm );
//FALCON_FUNC img_isGIF ( VMachine *vm );
FALCON_FUNC img_isJPG ( VMachine *vm );
//FALCON_FUNC img_isTIF ( VMachine *vm );
//FALCON_FUNC img_isPNG ( VMachine *vm );
//FALCON_FUNC img_isLBM ( VMachine *vm );


// Errors functions

FALCON_FUNC img_GetError ( VMachine *vm );
FALCON_FUNC img_SetError ( VMachine *vm );


}
}

#endif

/* end of sdlimage_ext.h */
