/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_ext.h

   The SDL True Type binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:11:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL True Type binding support module.
*/

#ifndef flc_sdlttf_ext_H
#define flc_sdlttf_ext_H

#include <falcon/setup.h>


namespace Falcon {
namespace Ext {

// Generic TTF
FALCON_FUNC ttf_Init( VMachine *vm );
FALCON_FUNC ttf_WasInit( VMachine *vm );
FALCON_FUNC ttf_InitAuto( VMachine *vm );
FALCON_FUNC ttf_Quit( VMachine *vm );
FALCON_FUNC ttf_Compiled_Version( VMachine *vm );
FALCON_FUNC ttf_Linked_Version( VMachine *vm );
FALCON_FUNC ttf_OpenFont( VMachine *vm );
FALCON_FUNC ttf_ByteSwappedUNICODE( VMachine *vm );

FALCON_FUNC ttf_GetFontStyle( VMachine *vm );
FALCON_FUNC ttf_SetFontStyle( VMachine *vm );
FALCON_FUNC ttf_FontHeight( VMachine *vm );
FALCON_FUNC ttf_FontAscent( VMachine *vm );
FALCON_FUNC ttf_FontDescent( VMachine *vm );
FALCON_FUNC ttf_FontLineSkip( VMachine *vm );
FALCON_FUNC ttf_FontFaces( VMachine *vm );
FALCON_FUNC ttf_FontFaceIsFixedWidth( VMachine *vm );
FALCON_FUNC ttf_FontFaceFamilyName( VMachine *vm );
FALCON_FUNC ttf_FontFaceStyleName( VMachine *vm );
FALCON_FUNC ttf_GlyphMetrics( VMachine *vm );
FALCON_FUNC ttf_SizeText( VMachine *vm );

FALCON_FUNC ttf_Render_Solid( VMachine *vm );
FALCON_FUNC ttf_Render_Shaded( VMachine *vm );
FALCON_FUNC ttf_Render_Blended( VMachine *vm );


}
}

#endif

/* end of sdlttf_ext.h */
