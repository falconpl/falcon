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
#include <falcon/suserdata.h>
#include <sdl_service.h>

namespace Falcon{
namespace Ext{

/** Automatic quit system. */
class QuitCarrier: public FalconData
{
public:
   QuitCarrier() {}
   virtual ~QuitCarrier();

   virtual void gcMark( VMachine* ) {}
   virtual FalconData* clone() const { return 0; }
};


/** Reflexive SDL Surface */
class SDLSurfaceCarrier_impl: public SDLSurfaceCarrier
{
   SDL_Surface *m_surface;


public:
   uint32 m_lockCount;

   SDLSurfaceCarrier_impl( SDL_Surface *s ):
      m_surface( s ),
      m_lockCount(0)
   {}

   virtual ~SDLSurfaceCarrier_impl();

   virtual void getProperty( VMachine *vm, const String &propName, Item &prop );
   virtual void setProperty(Falcon::VMachine*, const Falcon::String&, const Falcon::Item&);
   virtual void gcMark( VMachine* ) {}
   virtual FalconData* clone() const;
   virtual SDL_Surface* surface() const { return m_surface; }
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
   virtual void gcMark(VMachine*) {}
};

/** Reflexive SDL RWops */
class SDLRWopsCarrier
{
   SDL_RWops *m_rwops;


public:

   SDLRWopsCarrier( VMachine *vm );
   virtual ~SDLRWopsCarrier();

   //virtual void OpenFile();
   //virtual void OpenMem();
   //virtual void OpenCMem();
   //virtual void SetMemSpace();
   //virtual void FreeMemSpace();
   //virtual void Seek();
   //virtual void Tell();
   //virtual void Read();
   //virtual void Write();
   //virtual void Close();
   //virtual void getProperty( VMachine *vm, const String &propName, Item &prop );
   virtual SDL_RWops* data() const { return m_rwops; }
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
