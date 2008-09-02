/*
   FALCON - The Falcon Programming Language.
   FILE: sdl.cpp

   The SDL binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The sdl module - main file.
*/

extern "C" {
   #include <SDL.h>
}

#include <falcon/setup.h>
#include <falcon/enginedata.h>
#include <falcon/module.h>
#include <falcon/userdata.h>
#include "version.h"
#include "sdl_ext.h"
#include "sdl_service.h"

/*#
   @module sdl The SDL Falcon Module.
   @brief Main module for the Falcon SDL module suite.

   This is the base of the falcon SDL subsystem.
   The SDL library can be found at <a target="_new" href="http://www.libsdl.org/">http://www.libsdl.org</a>.

   @section Forewords

   The SDL Falcon module tries to stick with SDL interface and conventions
   whenever possible. However, in some cases where Falcon programming language
   provides structures and solutions that are better suited to perform certain
   tasks, the interface may diverge from the original one.

   The most significative case is the @a SDLSurface class and its derived classes
   that encapsulate many of the SDL operations that are menat to be performed
   on surfaces and screens.

   Also, event structures are not directly exposed to the Falcon programs; instead,
   they are reported through callbacks. In fact, callbacks and parameter expansion
   is several time faster than the creation of a Falcon object that should be then
   inspected and eventually marshalled.

   @beginmodule sdl
*/


Falcon::SDLService the_service;

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "sdl" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //=================================================================
   // Encapsulation SDL
   //
   Falcon::Symbol *c_sdl = self->addClass( "SDL" );
   self->addClassProperty( c_sdl, "INIT_VIDEO" ).setInteger( SDL_INIT_VIDEO );
   self->addClassProperty( c_sdl, "INIT_AUDIO" ).setInteger( SDL_INIT_AUDIO );
   self->addClassProperty( c_sdl, "INIT_TIMER" ).setInteger( SDL_INIT_TIMER );
   self->addClassProperty( c_sdl, "INIT_CDROM" ).setInteger( SDL_INIT_CDROM );
   self->addClassProperty( c_sdl, "INIT_JOYSTICK" ).setInteger( SDL_INIT_JOYSTICK );
   self->addClassProperty( c_sdl, "INIT_EVERYTHING" ).setInteger( SDL_INIT_EVERYTHING );
   self->addClassProperty( c_sdl, "INIT_NOPARACHUTE" ).setInteger( SDL_INIT_NOPARACHUTE );

   self->addClassProperty( c_sdl, "SWSURFACE" ).setInteger( SDL_SWSURFACE );
   self->addClassProperty( c_sdl, "HWSURFACE" ).setInteger( SDL_HWSURFACE );
   self->addClassProperty( c_sdl, "ASYNCBLIT" ).setInteger( SDL_ASYNCBLIT );
   self->addClassProperty( c_sdl, "ANYFORMAT" ).setInteger( SDL_ANYFORMAT );
   self->addClassProperty( c_sdl, "HWPALETTE" ).setInteger( SDL_HWPALETTE );
   self->addClassProperty( c_sdl, "DOUBLEBUF" ).setInteger( SDL_DOUBLEBUF );
   self->addClassProperty( c_sdl, "FULLSCREEN" ).setInteger( SDL_FULLSCREEN );
   self->addClassProperty( c_sdl, "OPENGL" ).setInteger( SDL_OPENGL );
   self->addClassProperty( c_sdl, "OPENGLBLIT" ).setInteger( SDL_OPENGLBLIT );
   self->addClassProperty( c_sdl, "RESIZABLE" ).setInteger( SDL_RESIZABLE );
   self->addClassProperty( c_sdl, "NOFRAME" ).setInteger( SDL_NOFRAME );
   self->addClassProperty( c_sdl, "HWACCEL" ).setInteger( SDL_HWACCEL );
   self->addClassProperty( c_sdl, "SRCCOLORKEY" ).setInteger( SDL_SRCCOLORKEY );
   self->addClassProperty( c_sdl, "RLEACCEL" ).setInteger( SDL_RLEACCEL );
   self->addClassProperty( c_sdl, "SRCALPHA" ).setInteger( SDL_SRCALPHA );
   self->addClassProperty( c_sdl, "PREALLOC" ).setInteger( SDL_PREALLOC );
   self->addClassProperty( c_sdl, "LOGPAL" ).setInteger( SDL_LOGPAL );
   self->addClassProperty( c_sdl, "PHYSPAL" ).setInteger( SDL_PHYSPAL );
   self->addClassProperty( c_sdl, "GRAB_QUERY" ).setInteger( SDL_GRAB_QUERY );
   self->addClassProperty( c_sdl, "GRAB_OFF" ).setInteger( SDL_GRAB_OFF );
   self->addClassProperty( c_sdl, "GRAB_ON" ).setInteger( SDL_GRAB_ON );
   self->addClassProperty( c_sdl, "ENABLE" ).setInteger( SDL_ENABLE );
   self->addClassProperty( c_sdl, "DISABLE" ).setInteger( SDL_DISABLE);
   self->addClassProperty( c_sdl, "QUERY" ).setInteger( SDL_QUERY );
   self->addClassProperty( c_sdl, "IGNORE" ).setInteger( SDL_IGNORE );

   self->addClassProperty( c_sdl, "APPMOUSEFOCUS" ).setInteger( SDL_APPMOUSEFOCUS );
   self->addClassProperty( c_sdl, "APPINPUTFOCUS" ).setInteger( SDL_APPINPUTFOCUS );
   self->addClassProperty( c_sdl, "APPACTIVE" ).setInteger( SDL_APPACTIVE );

   self->addClassProperty( c_sdl, "PRESSED" ).setInteger( SDL_PRESSED );
   self->addClassProperty( c_sdl, "RELEASED" ).setInteger( SDL_RELEASED );
   self->addClassProperty( c_sdl, "HAT_CENTERED" ).setInteger( SDL_HAT_CENTERED );
   self->addClassProperty( c_sdl, "HAT_UP" ).setInteger( SDL_HAT_UP );
   self->addClassProperty( c_sdl, "HAT_RIGHT" ).setInteger( SDL_HAT_RIGHT );
   self->addClassProperty( c_sdl, "HAT_DOWN" ).setInteger( SDL_HAT_DOWN );
   self->addClassProperty( c_sdl, "HAT_LEFT" ).setInteger( SDL_HAT_LEFT );
   self->addClassProperty( c_sdl, "HAT_RIGHTUP" ).setInteger( SDL_HAT_RIGHTUP );
   self->addClassProperty( c_sdl, "HAT_RIGHTDOWN" ).setInteger( SDL_HAT_RIGHTDOWN );
   self->addClassProperty( c_sdl, "HAT_LEFTUP" ).setInteger( SDL_HAT_LEFTUP );
   self->addClassProperty( c_sdl, "HAT_LEFTDOWN" ).setInteger( SDL_HAT_LEFTDOWN );

   self->addClassProperty( c_sdl, "BUTTON_LEFT" ).setInteger( SDL_BUTTON_LEFT );
   self->addClassProperty( c_sdl, "BUTTON_MIDDLE" ).setInteger( SDL_BUTTON_MIDDLE );
   self->addClassProperty( c_sdl, "BUTTON_RIGHT" ).setInteger( SDL_BUTTON_RIGHT );
   self->addClassProperty( c_sdl, "DEFAULT_REPEAT_DELAY" ).setInteger( SDL_DEFAULT_REPEAT_DELAY );
   self->addClassProperty( c_sdl, "DEFAULT_REPEAT_INTERVAL" ).setInteger( SDL_DEFAULT_REPEAT_INTERVAL );

   // Init and quit
   self->addClassMethod( c_sdl, "Init", Falcon::Ext::sdl_Init ).asSymbol()->
      addParam("flags");
   self->addClassMethod( c_sdl, "WasInit", Falcon::Ext::sdl_WasInit ).asSymbol()->
      addParam("flags");
   self->addClassMethod( c_sdl, "InitAuto", Falcon::Ext::sdl_InitAuto ).asSymbol()->
      addParam("flags");
   self->addClassMethod( c_sdl, "Quit", Falcon::Ext::sdl_Quit );
   self->addClassMethod( c_sdl, "QuitSubSystem", Falcon::Ext::sdl_QuitSubSystem ).asSymbol()->
      addParam("subsys");
   self->addClassMethod( c_sdl, "IsBigEndian", Falcon::Ext::sdl_IsBigEndian );

   // Generic video
   self->addClassMethod( c_sdl, "SetVideoMode", Falcon::Ext::sdl_SetVideoMode ).asSymbol()->
      addParam("width")->addParam("height")->addParam("bpp")->addParam("flags");
   self->addClassMethod( c_sdl, "GetVideoInfo", Falcon::Ext::sdl_GetVideoInfo );
   self->addClassMethod( c_sdl, "GetVideoSurface", Falcon::Ext::sdl_GetVideoSurface );
   self->addClassMethod( c_sdl, "VideoDriverName", Falcon::Ext::sdl_VideoDriverName );
   self->addClassMethod( c_sdl, "ListModes", Falcon::Ext::sdl_ListModes ).asSymbol()->
      addParam("format")->addParam("flags");
   self->addClassMethod( c_sdl, "VideoModeOK", Falcon::Ext::sdl_VideoModeOK ).asSymbol()->
      addParam("width")->addParam("height")->addParam("bpp")->addParam("flags");
   self->addClassMethod( c_sdl, "SetGamma", Falcon::Ext::sdl_SetGamma ).asSymbol()->
      addParam("red")->addParam("green")->addParam("blue");
   self->addClassMethod( c_sdl, "GetGammaRamp", Falcon::Ext::sdl_GetGammaRamp ).asSymbol()->
      addParam("aRet");
   self->addClassMethod( c_sdl, "SetGammaRamp", Falcon::Ext::sdl_SetGammaRamp ).asSymbol()->
      addParam("redbuf")->addParam("greenbuf")->addParam("bluebuf");
   self->addClassMethod( c_sdl, "CreateRGBSurface", Falcon::Ext::sdl_CreateRGBSurface ).asSymbol()->
      addParam("flags")->addParam("width")->addParam("height")->addParam("depth")->addParam("rMask")->addParam("gMask")->addParam("bMask")->addParam("aMask");
   self->addClassMethod( c_sdl, "CreateRGBSurfaceFrom", Falcon::Ext::sdl_CreateRGBSurfaceFrom ).asSymbol()->
      addParam("pixels")->addParam("width")->addParam("height")->addParam("depth")->addParam("rMask")->addParam("gMask")->addParam("bMask")->addParam("aMask");

   // WM
   self->addClassMethod( c_sdl, "WM_SetCaption", Falcon::Ext::sdl_WM_SetCaption ).asSymbol()->
      addParam("caption")->addParam("icon");
   self->addClassMethod( c_sdl, "WM_GetCaption", Falcon::Ext::sdl_WM_GetCaption );
   self->addClassMethod( c_sdl, "WM_IconifyWindow", Falcon::Ext::sdl_WM_IconifyWindow );
   self->addClassMethod( c_sdl, "WM_GrabInput", Falcon::Ext::sdl_WM_GrabInput ).asSymbol()->
      addParam("grab");

   // Cursor
   self->addClassMethod( c_sdl, "GetCursor", Falcon::Ext::sdl_GetCursor );
   self->addClassMethod( c_sdl, "ShowCursor", Falcon::Ext::sdl_ShowCursor ).asSymbol()->
      addParam("request");
   self->addClassMethod( c_sdl, "MakeCursor", Falcon::Ext::sdl_MakeCursor ).asSymbol()->
      addParam("aImage")->addParam("hotX")->addParam("hotY");
   self->addClassMethod( c_sdl, "CreateCursor", Falcon::Ext::sdl_CreateCursor ).asSymbol()->
      addParam("mbData")->addParam("mbMask")->addParam("width")->addParam("height")->addParam("Xspot")->addParam("Yspot");

   // Surface
   self->addClassMethod( c_sdl, "LoadBMP", Falcon::Ext::sdl_LoadBMP ).asSymbol()->
      addParam("filename");

   // Events
   self->addClassMethod( c_sdl, "PushEvent", Falcon::Ext::SDLEventHandler_PushEvent );
   self->addClassMethod( c_sdl, "PushUserEvent", Falcon::Ext::SDLEventHandler_PushUserEvent );
   self->addClassMethod( c_sdl, "PumpEvents", Falcon::Ext::sdl_PumpEvents);
   self->addClassMethod( c_sdl, "EventState", Falcon::Ext::sdl_EventState).asSymbol()->
      addParam("type")->addParam("state");
   self->addClassMethod( c_sdl, "GetKeyState", Falcon::Ext::sdl_GetKeyState);
   self->addClassMethod( c_sdl, "GetModState", Falcon::Ext::sdl_GetModState);
   self->addClassMethod( c_sdl, "SetModState", Falcon::Ext::sdl_SetModState).asSymbol()->
      addParam("state");
   self->addClassMethod( c_sdl, "GetKeyName", Falcon::Ext::sdl_GetKeyName).asSymbol()->
      addParam("key");
   self->addClassMethod( c_sdl, "EnableUNICODE", Falcon::Ext::sdl_EnableUNICODE).asSymbol()->
      addParam("mode");
   self->addClassMethod( c_sdl, "EnableKeyRepeat", Falcon::Ext::sdl_EnableKeyRepeat).asSymbol()->
      addParam("delay")->addParam("interval");
   self->addClassMethod( c_sdl, "GetAppState", Falcon::Ext::sdl_GetAppState);
   self->addClassMethod( c_sdl, "JoystickEventState", Falcon::Ext::sdl_JoystickEventState).asSymbol()->
      addParam("mode");
   self->addClassMethod( c_sdl, "JoystickUpdate", Falcon::Ext::sdl_JoystickUpdate);

    //============================================================
   // SDL rectangle class
   //
   Falcon::Symbol *c_rect = self->addClass( "SDLRect", Falcon::Ext::SDLRect_init );
   c_rect->setWKS( true );
   self->addClassProperty( c_rect, "w" );
   self->addClassProperty( c_rect, "h" );
   self->addClassProperty( c_rect, "x" );
   self->addClassProperty( c_rect, "y" );

   //============================================================
   // SDL Surface class
   //
   Falcon::Symbol *c_surface = self->addClass( "SDLSurface" );
   c_surface->getClassDef()->setObjectManager( &Falcon::core_user_data_manager_cacheful );
   c_surface->setWKS( true );
   self->addClassProperty( c_surface, "w" );
   self->addClassProperty( c_surface, "h" );
   self->addClassProperty( c_surface, "flags" );
   self->addClassProperty( c_surface, "pitch" );
   self->addClassProperty( c_surface, "clip_rect" );
   self->addClassProperty( c_surface, "pixels" );
   self->addClassProperty( c_surface, "bpp" );
   self->addClassProperty( c_surface, "format" );

   self->addClassMethod( c_surface, "BlitSurface", Falcon::Ext::SDLSurface_BlitSurface ).asSymbol()->
      addParam("srcRect")->addParam("dest")->addParam("dstRect");
   self->addClassMethod( c_surface, "SaveBMP", Falcon::Ext::SDLSurface_SaveBMP ).asSymbol()->
      addParam("filename");
   self->addClassMethod( c_surface, "SetPixel", Falcon::Ext::SDLSurface_SetPixel ).asSymbol()->
      addParam("x")->addParam("y")->addParam("value");
   self->addClassMethod( c_surface, "GetPixel", Falcon::Ext::SDLSurface_GetPixel ).asSymbol()->
      addParam("x")->addParam("y");
   self->addClassMethod( c_surface, "GetPixelIndex", Falcon::Ext::SDLSurface_GetPixelIndex ).asSymbol()->
      addParam("x")->addParam("y");
   self->addClassMethod( c_surface, "LockSurface", Falcon::Ext::SDLSurface_LockSurface );
   self->addClassMethod( c_surface, "UnlockSurface", Falcon::Ext::SDLSurface_UnlockSurface );
   self->addClassMethod( c_surface, "LockIfNeeded", Falcon::Ext::SDLSurface_LockIfNeeded );
   self->addClassMethod( c_surface, "UnlockIfNeeded", Falcon::Ext::SDLSurface_UnlockIfNeeded );
   self->addClassMethod( c_surface, "IsLockNeeded", Falcon::Ext::SDLSurface_IsLockNeeded );
   self->addClassMethod( c_surface, "FillRect", Falcon::Ext::SDLSurface_FillRect ).asSymbol()->
      addParam("rect")->addParam("color");
   self->addClassMethod( c_surface, "GetRGBA", Falcon::Ext::SDLSurface_GetRGBA ).asSymbol()->
      addParam("color")->addParam("retArray");
   self->addClassMethod( c_surface, "MapRGBA", Falcon::Ext::SDLSurface_MapRGBA ).asSymbol()->
      addParam("red")->addParam("green")->addParam("blue")->addParam("alpha");
   self->addClassMethod( c_surface, "SetColors", Falcon::Ext::SDLSurface_SetColors ).asSymbol()->
      addParam("colors")->addParam("firstColor");
   self->addClassMethod( c_surface, "SetIcon", Falcon::Ext::SDLSurface_SetIcon );

   //============================================================
   // SDL Pixel Format
   //
   Falcon::Symbol *c_pixf = self->addClass( "SDLPixelFormat" );
   c_pixf->setWKS( true );
   self->addClassProperty( c_pixf, "palette" );
   self->addClassProperty( c_pixf, "BitsPerPixel" );
   self->addClassProperty( c_pixf, "BytesPerPixel" );
   self->addClassProperty( c_pixf, "Rloss" );
   self->addClassProperty( c_pixf, "Gloss" );
   self->addClassProperty( c_pixf, "Bloss" );
   self->addClassProperty( c_pixf, "Aloss" );
   self->addClassProperty( c_pixf, "Rshift" );
   self->addClassProperty( c_pixf, "Gshift" );
   self->addClassProperty( c_pixf, "Bshift" );
   self->addClassProperty( c_pixf, "Ashift" );
   self->addClassProperty( c_pixf, "Rmask" );
   self->addClassProperty( c_pixf, "Gmask" );
   self->addClassProperty( c_pixf, "Bmask" );
   self->addClassProperty( c_pixf, "Amask" );
   self->addClassProperty( c_pixf, "colorkey" );
   self->addClassProperty( c_pixf, "alpha" );

   //============================================================
   // SDL Video Info
   //
   Falcon::Symbol *c_vi = self->addClass( "SDLVideoInfo" );
   c_vi->setWKS( true );
   self->addClassProperty( c_vi, "hw_available" );
   self->addClassProperty( c_vi, "wm_available" );
   self->addClassProperty( c_vi, "blit_hw" );
   self->addClassProperty( c_vi, "blit_hw_CC" );
   self->addClassProperty( c_vi, "blit_hw_A" );
   self->addClassProperty( c_vi, "blit_sw" );
   self->addClassProperty( c_vi, "blit_sw_CC" );
   self->addClassProperty( c_vi, "blit_sw_A" );
   self->addClassProperty( c_vi, "blit_fill" );
   self->addClassProperty( c_vi, "video_mem" );
   self->addClassProperty( c_vi, "vfmt" );

   //============================================================
   // SDL Palette Format
   //
   Falcon::Symbol *c_palette = self->addClass( "SDLPalette" );
   c_palette->setWKS( true );
   self->addClassProperty( c_palette, "ncolors" );
   self->addClassProperty( c_palette, "colors" );
   self->addClassMethod( c_palette, "GetColor", Falcon::Ext::SDLPalette_getColor ).asSymbol()->
      addParam("colorIndex")->addParam("colArray");
   self->addClassMethod( c_palette, "SetColor", Falcon::Ext::SDLPalette_setColor ).asSymbol()->
      addParam("colorIndex")->addParam("red")->addParam("green")->addParam("blue");

   //============================================================
   // SDL Palette Format
   //
   /*#
      @class SDLColor
      @brief Storage for RGB values

      @prop r red value
      @prop g green value
      @prop b blue value
   */

   Falcon::Symbol *c_sdlcolor = self->addClass( "SDLColor", Falcon::Ext::SDLColor_init );
   c_sdlcolor->setWKS( true );
   SDL_Color sdl_color;
   self->addClassProperty( c_sdlcolor, "r" )
         .setReflective( Falcon::e_reflectByte, &sdl_color, &sdl_color.r );
   self->addClassProperty( c_sdlcolor, "g" )
         .setReflective( Falcon::e_reflectByte, &sdl_color, &sdl_color.g );
   self->addClassProperty( c_sdlcolor, "b" )
         .setReflective( Falcon::e_reflectByte, &sdl_color, &sdl_color.b );

   //============================================================
   // SDL Cursor
   //
   Falcon::Symbol *c_cursor = self->addClass( "SDLCursor", false ); // not instantiable
   c_cursor->setWKS( true );
   c_cursor->getClassDef()->setObjectManager( &Falcon::core_falcon_data_manager );
   self->addClassMethod( c_cursor, "SetCursor", Falcon::Ext::SDLCursor_SetCursor );

   //============================================================
   // SDL screen class
   //
   Falcon::Symbol *c_screen = self->addClass( "SDLScreen" );
   c_screen->setWKS( true );
   c_screen->getClassDef()->addInheritance( new Falcon::InheritDef( c_surface ) );
   self->addClassMethod( c_screen, "UpdateRect", Falcon::Ext::SDLScreen_UpdateRect ).asSymbol()->
      addParam("xOrRect")->addParam("y")->addParam("width")->addParam("height");
   self->addClassMethod( c_screen, "UpdateRects", Falcon::Ext::SDLScreen_UpdateRects ).asSymbol()->
      addParam("aRects");
   self->addClassMethod( c_screen, "Flip", Falcon::Ext::SDLScreen_Flip );
   self->addClassMethod( c_screen, "SetPalette", Falcon::Ext::SDLScreen_SetPalette ).asSymbol()->
      addParam("flags")->addParam("colors")->addParam("firstColor");
   self->addClassMethod( c_screen, "ToggleFullScreen", Falcon::Ext::SDLScreen_ToggleFullScreen );

   //============================================================
   // Event subsystem
   //
   Falcon::Ext::declare_events( self );

   //============================================================
   // SDL Error class
   //
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *sdlerr_cls = self->addClass( "SDLError", Falcon::Ext::SDLError_init );
   sdlerr_cls->setWKS( true );
   sdlerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   //=================================================================
   // Auto quit feature
   //
   Falcon::Symbol *c_sdl_aq = self->addClass( "_SDL_AutoQuit" );
   c_sdl_aq->setWKS( true );
   c_sdl_aq->exported( false );
   c_sdl_aq->getClassDef()->setObjectManager( &Falcon::core_falcon_data_manager );
   self->addClassMethod( c_sdl_aq, "Quit", Falcon::Ext::sdl_Quit );

   //==================================================================
   // Service feature.
   //
   self->publishService( &the_service );

   return self;
}

/* end of sdl.cpp */

