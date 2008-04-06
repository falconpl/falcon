/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_mod.h

   The SDL binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL binding support module - module specific extensions.
*/

#ifndef FALCON_SDL_MOD
#define FALCON_SDL_MOD

#include <falcon/setup.h>
#include <falcon/suserdata.h>
#include <falcon/error.h>

#include <SDL.h>

#define FALCON_SDL_ERROR_BASE 2100

namespace Falcon{
namespace Ext{

/** Automatic quit system. */
class QuitCarrier: public UserData
{
public:
   QuitCarrier() {}
   ~QuitCarrier();
};


/** Low level SDL error */
class SDLError: public ::Falcon::Error
{
public:
   SDLError():
      Error( "SDLError" )
   {}

   SDLError( const ErrorParam &params  ):
      Error( "SDLError", params )
      {}
};

/** Reflexive SDL Surface */
class SDLSurfaceCarrier: public SharedUserData
{
public:
   SDL_Surface *m_surface;
   uint32 m_lockCount;

   SDLSurfaceCarrier( VMachine *vm, SDL_Surface *s ):
      SharedUserData( vm ),
      m_surface( s ),
      m_lockCount(0)
   {}

   ~SDLSurfaceCarrier();

   virtual bool isReflective() const;
   virtual void getProperty( VMachine *vm, const String &propName, Item &prop );
   virtual void setProperty( VMachine *vm, const String &propName, Item &prop );
   virtual UserData *clone() const;
};

/** Opaque Cursor structure carrier */
class SDLCursorCarrier: public UserData
{
public:
   SDL_Cursor *m_cursor;
   bool m_bCreated;

   SDLCursorCarrier( SDL_Cursor *cursor, bool bCreated = true ):
      m_cursor( cursor ),
      m_bCreated( bCreated )
   {}

   ~SDLCursorCarrier();
};

//==========================================
// Utilities
//

bool RectToObject( const ::SDL_Rect &rect, CoreObject *obj );
bool ObjectToRect( CoreObject *obj, ::SDL_Rect &rect );
CoreObject *MakeRectInst( VMachine *vm, const ::SDL_Rect &rect );
CoreObject *MakePixelFormatInst( VMachine *vm, SDLSurfaceCarrier *carrier, ::SDL_PixelFormat *fmt = 0 );
bool ObjectToPixelFormat( CoreObject *obj, ::SDL_PixelFormat *fmt );
CoreObject *MakeVideoInfo( VMachine *vm, const ::SDL_VideoInfo *info );

typedef struct tag_sdl_mouse_state
{
   int state;
   int x, y;
   int xrel, yrel;
} sdl_mouse_state;

}
}
#endif

/* end of sdl_mod.h */
