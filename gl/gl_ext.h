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
   FALCON_FUNC opengl_Viewport( VMachine *vm );
   FALCON_FUNC sdlgl_SetAttribute( VMachine *vm );
   FALCON_FUNC sdlgl_GetAttribute( VMachine *vm );
   FALCON_FUNC opengl_LoadIdentity( VMachine *vm );
   FALCON_FUNC opengl_MatrixMode( VMachine *vm );
   FALCON_FUNC opengl_ClearColor( VMachine *vm );
   FALCON_FUNC opengl_ClearDepth( VMachine *vm );
   FALCON_FUNC opengl_Enable( VMachine *vm );
   FALCON_FUNC opengl_Clear( VMachine *vm );
   FALCON_FUNC opengl_Begin( VMachine *vm );
   FALCON_FUNC opengl_End( VMachine *vm );
   FALCON_FUNC opengl_SwapBuffers( VMachine *vm );
   FALCON_FUNC opengl_Color3d( VMachine *vm );
   FALCON_FUNC opengl_Vertex3d( VMachine *vm );
   FALCON_FUNC opengl_Translate( VMachine *vm );
   FALCON_FUNC opengl_Rotate( VMachine *vm );
   FALCON_FUNC opengl_GenTextures( VMachine *vm );
   FALCON_FUNC opengl_ShadeModel( VMachine *vm );
   FALCON_FUNC opengl_Hint( VMachine *vm );
   FALCON_FUNC opengl_DepthFunc( VMachine *vm );
   FALCON_FUNC opengl_TexCoord2d( VMachine *vm );
   FALCON_FUNC opengl_BindTexture( VMachine *vm );
   FALCON_FUNC opengl_TexParameteri( VMachine *vm );
   FALCON_FUNC opengl_TexImage2D( VMachine *vm );

}
}

#endif

/* end of sdlopengl_ext.h */
