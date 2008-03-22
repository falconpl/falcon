/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_ext.h

   The SDL binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100
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
   The SDL binding support module.
*/

#ifndef flc_sdl_ext_H
#define flc_sdl_ext_H

#include <falcon/setup.h>


namespace Falcon {
namespace Ext {

// Init and quit
FALCON_FUNC sdl_Init( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_WasInit( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_InitAuto( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_Quit( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_QuitSubSystem( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_IsBigEndian( ::Falcon::VMachine *vm );

// Generic video
FALCON_FUNC sdl_SetVideoMode( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_GetVideoInfo( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_VideoDriverName( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_GetVideoSurface ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_ListModes ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_VideoModeOK ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_SetGamma ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_GetGammaRamp ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_SetGammaRamp ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_CreateRGBSurface ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_CreateRGBSurfaceFrom ( ::Falcon::VMachine *vm );

// WM
FALCON_FUNC sdl_WM_SetCaption ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_WM_GetCaption ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_WM_IconifyWindow ( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_WM_GrabInput ( ::Falcon::VMachine *vm );


// Surface video
FALCON_FUNC sdl_LoadBMP( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_BlitSurface( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_SaveBMP( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_SetPixel( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_GetPixel( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_GetPixelIndex( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_LockSurface( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_UnlockSurface( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_LockIfNeeded( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_UnlockIfNeeded( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_IsLockNeeded( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_FillRect( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_SetColors( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_SetIcon ( ::Falcon::VMachine *vm );

// PixelFormat
FALCON_FUNC SDLSurface_GetRGBA( ::Falcon::VMachine *vm );
FALCON_FUNC SDLSurface_MapRGBA( ::Falcon::VMachine *vm );

// Rectangle
FALCON_FUNC SDLRect_init( ::Falcon::VMachine *vm );

// Screen video
FALCON_FUNC SDLScreen_UpdateRect( ::Falcon::VMachine *vm );
FALCON_FUNC SDLScreen_UpdateRects( ::Falcon::VMachine *vm );
FALCON_FUNC SDLScreen_Flip( ::Falcon::VMachine *vm );
FALCON_FUNC SDLScreen_SetPalette( ::Falcon::VMachine *vm );
FALCON_FUNC SDLScreen_ToggleFullScreen( ::Falcon::VMachine *vm );

// Cursor
FALCON_FUNC sdl_GetCursor( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_ShowCursor( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_MakeCursor( ::Falcon::VMachine *vm );
FALCON_FUNC sdl_CreateCursor( ::Falcon::VMachine *vm );

FALCON_FUNC SDLCursor_SetCursor( ::Falcon::VMachine *vm );

// Palette
FALCON_FUNC SDLPalette_getColor( ::Falcon::VMachine *vm );
FALCON_FUNC SDLPalette_setColor( ::Falcon::VMachine *vm );

// Error
FALCON_FUNC  SDLError_init ( ::Falcon::VMachine *vm );

}
}

#endif

/* end of sdl_ext.h */
