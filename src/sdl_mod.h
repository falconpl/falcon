/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_mod.h

   The SDL binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100

   Last modified because:
   Federico Baroni - Thu 9 Oct 2008 23:08:41 - RWops carrier class added

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
#include <falcon/falcondata.h>
#include <sdl_service.h>


namespace Falcon{
namespace Ext{

//====================================================
// Reflectors
//

// surface
CoreObject* SDLSurface_Factory( const CoreClass *cls, void *user_data, bool );
void sdl_surface_bpp_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& );
void sdl_surface_pixels_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& );
void sdl_surface_format_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& );
void sdl_surface_clip_rect_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& );

// rect
CoreObject* SDLRect_Factory( const CoreClass *cls, void *user_data, bool );
CoreObject* SDLColor_Factory( const CoreClass *cls, void *user_data, bool );

/** Automatic quit system. */
class QuitCarrier: public FalconData
{
public:
   QuitCarrier() {}
   virtual ~QuitCarrier();

   virtual void gcMark( uint32 ) {}
   virtual FalconData* clone() const { return 0; }
};


/** Reflexive SDL Surface */
class SDLSurfaceCarrier_impl: public SDLSurfaceCarrier
{
   MemBuf* m_mbPixelCache;

public:
   uint32 m_lockCount;

   SDLSurfaceCarrier_impl( const CoreClass* cls, SDL_Surface *s );
   virtual ~SDLSurfaceCarrier_impl();

   virtual void gcMark( uint32 );
   virtual CoreObject* clone() const;
   virtual SDL_Surface* surface() const { return (SDL_Surface*) getUserData(); }

   void setPixelCache( MemBuf* mb );

   friend void sdl_surface_pixels_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& );
};


class SDLRectCarrier: public ReflectObject
{
public:
   SDLRectCarrier( const CoreClass* cls, SDL_Rect *r ):
      ReflectObject( cls, r )
   {}

   virtual ~SDLRectCarrier();
   virtual void gcMark( uint32 ) {}
   virtual CoreObject* clone() const;
   virtual SDL_Rect* rect() const { return (SDL_Rect*) getUserData(); }
};


class SDLColorCarrier: public ReflectObject
{
public:
   SDLColorCarrier( const CoreClass* cls, SDL_Color *c ):
      ReflectObject( cls, c )
   {}

   virtual ~SDLColorCarrier();
   virtual void gcMark( uint32 ) {}
   virtual CoreObject* clone() const;
   virtual SDL_Color* color() const { return (SDL_Color*) getUserData(); }
};

/** Opaque Cursor structure carrier */
class SDLCursorCarrier: public FalconData
{
public:
   SDL_Cursor *m_cursor;
   bool m_bCreated;

   SDLCursorCarrier( SDL_Cursor *cursor, bool bCreated = true ):
      m_cursor( cursor ),
      m_bCreated( bCreated )
   {}

   virtual ~SDLCursorCarrier();
   virtual FalconData *clone() const { return 0; }
   virtual void gcMark(uint32) {}
};



//==========================================
// Mouse state
//

typedef struct tag_sdl_mouse_state
{
   int state;
   int x, y;
   int xrel, yrel;
} sdl_mouse_state;

class Inst_SdlMouseState: public CRObject
{
public:
   sdl_mouse_state m_ms;

   Inst_SdlMouseState( const CoreClass* gen ):
      CRObject( gen )
   {
      setUserData( &m_ms );
   }

   Inst_SdlMouseState( const Inst_SdlMouseState &other ):
      CRObject( other.m_generatedBy )
   {
      m_ms = other.m_ms;
      setUserData( &m_ms );
   }

   virtual CoreObject *clone() const { new Inst_SdlMouseState( *this ); }
   virtual void gcMark(VMachine*) {}
};

CoreObject* SdlMouseState_Factory( const CoreClass *cls, void *, bool );

//==========================================
// Utilities
//

CoreObject *MakeRectInst( VMachine *vm, const ::SDL_Rect &rect );
CoreObject *MakePixelFormatInst( VMachine *vm, SDLSurfaceCarrier *carrier, ::SDL_PixelFormat *fmt = 0 );
bool ObjectToPixelFormat( CoreObject *obj, ::SDL_PixelFormat *fmt );
CoreObject *MakeVideoInfo( VMachine *vm, const ::SDL_VideoInfo *info );

}
}
#endif

/* end of sdl_mod.h */
