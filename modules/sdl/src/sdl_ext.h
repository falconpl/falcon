/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_ext.h

   The SDL binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
FALCON_FUNC sdl_Init( VMachine *vm );
FALCON_FUNC sdl_WasInit( VMachine *vm );
FALCON_FUNC sdl_InitAuto( VMachine *vm );
FALCON_FUNC sdl_Quit( VMachine *vm );
FALCON_FUNC sdl_QuitSubSystem( VMachine *vm );
FALCON_FUNC sdl_IsBigEndian( VMachine *vm );

// Generic video
FALCON_FUNC sdl_SetVideoMode( VMachine *vm );
FALCON_FUNC sdl_GetVideoInfo( VMachine *vm );
FALCON_FUNC sdl_VideoDriverName( VMachine *vm );
FALCON_FUNC sdl_GetVideoSurface ( VMachine *vm );
FALCON_FUNC sdl_ListModes ( VMachine *vm );
FALCON_FUNC sdl_VideoModeOK ( VMachine *vm );
FALCON_FUNC sdl_SetGamma ( VMachine *vm );
FALCON_FUNC sdl_GetGammaRamp ( VMachine *vm );
FALCON_FUNC sdl_SetGammaRamp ( VMachine *vm );
FALCON_FUNC sdl_CreateRGBSurface ( VMachine *vm );
FALCON_FUNC sdl_CreateRGBSurfaceFrom ( VMachine *vm );

// WM
FALCON_FUNC sdl_WM_SetCaption ( VMachine *vm );
FALCON_FUNC sdl_WM_GetCaption ( VMachine *vm );
FALCON_FUNC sdl_WM_IconifyWindow ( VMachine *vm );
FALCON_FUNC sdl_WM_GrabInput ( VMachine *vm );


// Surface video
FALCON_FUNC sdl_LoadBMP( VMachine *vm );
FALCON_FUNC SDLSurface_BlitSurface( VMachine *vm );
FALCON_FUNC SDLSurface_SaveBMP( VMachine *vm );
FALCON_FUNC SDLSurface_SetPixel( VMachine *vm );
FALCON_FUNC SDLSurface_GetPixel( VMachine *vm );
FALCON_FUNC SDLSurface_GetPixelIndex( VMachine *vm );
FALCON_FUNC SDLSurface_LockSurface( VMachine *vm );
FALCON_FUNC SDLSurface_UnlockSurface( VMachine *vm );
FALCON_FUNC SDLSurface_LockIfNeeded( VMachine *vm );
FALCON_FUNC SDLSurface_UnlockIfNeeded( VMachine *vm );
FALCON_FUNC SDLSurface_IsLockNeeded( VMachine *vm );
FALCON_FUNC SDLSurface_FillRect( VMachine *vm );
FALCON_FUNC SDLSurface_SetColors( VMachine *vm );
FALCON_FUNC SDLSurface_SetIcon ( VMachine *vm );
FALCON_FUNC SDLSurface_SetAlpha ( VMachine *vm );
FALCON_FUNC SDLSurface_DisplayFormatAlpha( VMachine *vm );

// PixelFormat
FALCON_FUNC SDLSurface_GetRGBA( VMachine *vm );
FALCON_FUNC SDLSurface_MapRGBA( VMachine *vm );

// Rectangle
FALCON_FUNC SDLRect_init( VMachine *vm );

// Screen video
FALCON_FUNC SDLScreen_UpdateRect( VMachine *vm );
FALCON_FUNC SDLScreen_UpdateRects( VMachine *vm );
FALCON_FUNC SDLScreen_Flip( VMachine *vm );
FALCON_FUNC SDLScreen_SetPalette( VMachine *vm );
FALCON_FUNC SDLScreen_ToggleFullScreen( VMachine *vm );

// Cursor
FALCON_FUNC sdl_GetCursor( VMachine *vm );
FALCON_FUNC sdl_ShowCursor( VMachine *vm );
FALCON_FUNC sdl_MakeCursor( VMachine *vm );
FALCON_FUNC sdl_CreateCursor( VMachine *vm );

FALCON_FUNC SDLCursor_SetCursor( VMachine *vm );

// Palette
FALCON_FUNC SDLPalette_getColor( VMachine *vm );
FALCON_FUNC SDLPalette_setColor( VMachine *vm );

// Error
FALCON_FUNC  SDLError_init ( VMachine *vm );

// Events
void declare_events( Module *self );
FALCON_FUNC sdl_PollEvent( VMachine *vm );
FALCON_FUNC sdl_WaitEvent( VMachine *vm );
FALCON_FUNC sdl_StartEvents( VMachine *vm );
FALCON_FUNC sdl_StopEvents( VMachine *vm );

FALCON_FUNC sdl_PumpEvents( VMachine *vm );
FALCON_FUNC sdl_EventState( VMachine *vm );
FALCON_FUNC sdl_GetKeyState( VMachine *vm );
FALCON_FUNC sdl_GetModState( VMachine *vm );
FALCON_FUNC sdl_SetModState( VMachine *vm );
FALCON_FUNC sdl_GetKeyName( VMachine *vm );
FALCON_FUNC sdl_EnableUNICODE( VMachine *vm );
FALCON_FUNC sdl_EnableKeyRepeat( VMachine *vm );
FALCON_FUNC sdl_GetAppState( VMachine *vm );
FALCON_FUNC sdl_JoystickEventState( VMachine *vm );
FALCON_FUNC sdl_JoystickUpdate( VMachine *vm );

// class SDLMouseState
FALCON_FUNC SDLMouseState_init( VMachine *vm );
FALCON_FUNC SDLMouseState_Refresh( VMachine *vm );
FALCON_FUNC SDLMouseState_PumpAndRefresh( VMachine *vm );

// class SDLColor
FALCON_FUNC SDLColor_init( VMachine *vm );

}
}

#endif

/* end of sdl_ext.h */
