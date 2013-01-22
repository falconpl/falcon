/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_mod.cpp

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
#define FALCON_EXPORT_SERVICE
#include <falcon/vm.h>
#include <falcon/membuf.h>
#include <falcon/stream.h>
#include "sdl_mod.h"

extern "C" {
   #include <SDL.h>
}

/*#
   @beginmodule sdl
*/


namespace Falcon {
namespace Ext {

//====================================================
// Module initialization.
//

SDLEventListener* s_EvtListener = 0;
Mutex* s_mtx_events = 0;

FALCON_SERVICE SDLModule::SDLModule()
{
   s_mtx_events = new Mutex;
   s_EvtListener = 0; // to be sure
}


FALCON_SERVICE SDLModule::~SDLModule()
{
   SDLEventListener *evt;
   s_mtx_events->lock();
   evt = s_EvtListener;
   s_EvtListener = 0;
   s_mtx_events->unlock();

   if( evt != 0 )
      evt->stop();
   delete s_mtx_events;
}

//=======================================
// Surface reflection
//

SDLSurfaceCarrier_impl::SDLSurfaceCarrier_impl( const CoreClass* cls, SDL_Surface *s ):
      SDLSurfaceCarrier( cls ),
      m_mbPixelCache( 0 ),
      m_lockCount(0)
{
   if( s != 0 )
   {
      if ( s->refcount == 1 )
         gcMemAccount( s->h * s->w * s->format->BytesPerPixel );

      s->refcount++;
   }

   setUserData( s );
}

SDLSurfaceCarrier_impl::~SDLSurfaceCarrier_impl()
{
   if ( surface() != 0 )
   {
      while( m_lockCount > 0 )
      {
         m_lockCount--;
         SDL_UnlockSurface( surface() );
      }

      if ( surface()->refcount == 1 )
         gcMemUnaccount( surface()->h * surface()->w * surface()->format->BytesPerPixel );

      SDL_FreeSurface( surface() );
   }
}


SDLSurfaceCarrier_impl *SDLSurfaceCarrier_impl::clone() const
{
   return new SDLSurfaceCarrier_impl( generator(), surface() );
}

bool SDLSurfaceCarrier_impl::deserialize( Stream *stream, bool bLive )
{
   CacheObject::deserialize(stream,bLive);
   if( bLive )
   {
      SDL_Surface* s;
      s = (SDL_Surface*) m_user_data;

      if ( s->refcount == 1 )
              gcMemAccount( s->h * s->w * s->format->BytesPerPixel );

      s->refcount++;
      return true;
   }

   return false;
}


void SDLSurfaceCarrier_impl::setPixelCache( MemBuf *mb )
{
   m_mbPixelCache = mb;
   mb->dependant( this );
}

void SDLSurfaceCarrier_impl::gcMark( uint32 gen )
{
   if ( m_mbPixelCache != 0 )
      m_mbPixelCache->mark( gen );
}

CoreObject* SDLSurface_Factory( const CoreClass *cls, void *user_data, bool )
{
   return new SDLSurfaceCarrier_impl( cls, (SDL_Surface*) user_data );
}

void sdl_surface_bpp_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& )
{
   SDL_Surface* surf = (SDL_Surface*) user_data;
   property = (int64) surf->format->BytesPerPixel;
}

void sdl_surface_pixels_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& )
{
   SDLSurfaceCarrier_impl* self = dyncast<SDLSurfaceCarrier_impl *>(co);

   if ( self->m_mbPixelCache != 0 )
   {
      property = self->m_mbPixelCache;
      return;
   }

   SDL_Surface* surf = (SDL_Surface*) user_data;
   fassert( surf != 0 );
   MemBuf *mb;

   switch( surf->format->BytesPerPixel )
   {
      case 1: mb = new MemBuf_1( (byte*)surf->pixels, surf->h * surf->w, 0 ); break;
      case 2: mb = new MemBuf_2( (byte*)surf->pixels, surf->h * surf->w, 0 ); break;
      case 3: mb = new MemBuf_3( (byte*)surf->pixels, surf->h * surf->w, 0 ); break;
      case 4: mb = new MemBuf_4( (byte*)surf->pixels, surf->h * surf->w, 0 ); break;
      default:
         fassert( false );
         return;
   }

   self->setPixelCache( mb );
   property = mb;
}

void sdl_surface_format_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& )
{
   property = MakePixelFormatInst( VMachine::getCurrent(), dyncast<SDLSurfaceCarrier*>(co) );
}

void sdl_surface_clip_rect_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& )
{
   SDL_Surface* surf = (SDL_Surface*) user_data;
   property = MakeRectInst( VMachine::getCurrent(), surf->clip_rect );
}

//=================================================
// Rect Carrier
//
CoreObject* SDLRect_Factory( const CoreClass *cls, void *user_data, bool )
{
   return new SDLRectCarrier( cls, (SDL_Rect*) user_data );
}

SDLRectCarrier::~SDLRectCarrier()
{
   free( rect() );
}

SDLRectCarrier* SDLRectCarrier::clone() const
{
   SDL_Rect* r = (SDL_Rect*) malloc( sizeof( SDL_Rect ) );
   *r = *rect();
   return new SDLRectCarrier( generator(), r );
}


//=================================================
// Rect Carrier
//

CoreObject* SDLColor_Factory( const CoreClass *cls, void *user_data, bool )
{
   return new SDLColorCarrier( cls, (SDL_Color*) user_data );
}

SDLColorCarrier::~SDLColorCarrier()
{
   free( color() );
}

SDLColorCarrier* SDLColorCarrier::clone() const
{
   SDL_Color* c = (SDL_Color*) malloc( sizeof( SDL_Color ) );
   *c = *color();
   return new SDLColorCarrier( generator(), c );
}

//=======================================
// Mouse state
//

CoreObject* SdlMouseState_Factory( const CoreClass *cls, void *, bool )
{
   return new Inst_SdlMouseState( cls );
}

//=======================================
// Quit carrier
//
QuitCarrier::~QuitCarrier()
{
   SDL_Quit();
}


//==========================================
// Cusror carrier
//

SDLCursorCarrier::~SDLCursorCarrier()
{
   if ( m_bCreated )
      SDL_FreeCursor( m_cursor );
}


//==========================================
// Utilities
//

CoreObject *MakeRectInst( VMachine *vm, const ::SDL_Rect &rect )
{
   Item *cls = vm->findWKI( "SDLRect" );
   fassert( cls != 0 );
   SDL_Rect* nrect = (SDL_Rect*) malloc( sizeof( SDL_Rect ) );
   memcpy( nrect, &rect, sizeof( SDL_Rect ) );

   CoreObject *obj = cls->asClass()->createInstance( nrect );
   return obj;
}

CoreObject *MakePixelFormatInst( VMachine *vm, SDLSurfaceCarrier *carrier, ::SDL_PixelFormat *fmt )
{
   Item *cls = vm->findWKI( "SDLPixelFormat" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();

   if ( carrier != 0 )
      fmt = carrier->surface()->format;

   obj->setProperty( "BitsPerPixel", (int64) fmt->BitsPerPixel );
   obj->setProperty( "BytesPerPixel", (int64) fmt->BytesPerPixel );
   obj->setProperty( "Rloss", (int64) fmt->Rloss );
   obj->setProperty( "Gloss", (int64) fmt->Gloss );
   obj->setProperty( "Bloss", (int64) fmt->Bloss );
   obj->setProperty( "Aloss", (int64) fmt->Aloss );

   obj->setProperty( "Rshift", (int64) fmt->Rshift );
   obj->setProperty( "Gshift", (int64) fmt->Gshift );
   obj->setProperty( "Bshift", (int64) fmt->Bshift );
   obj->setProperty( "Ashift", (int64) fmt->Ashift );

   obj->setProperty( "Rmask", (int64) fmt->Rmask );
   obj->setProperty( "Gmask", (int64) fmt->Gmask );
   obj->setProperty( "Bmask", (int64) fmt->Bmask );
   obj->setProperty( "Amask", (int64) fmt->Amask );
   obj->setProperty( "colorkey", (int64) fmt->colorkey );
   obj->setProperty( "alpha", (int64) fmt->alpha );

   // If we have a palette, creaete an object of class palette and stuff it in.
   if( fmt->palette != 0 )
   {
      Item *clspal = vm->findWKI( "SDLPalette" );
      fassert( clspal != 0 );
      CoreObject *objpal = clspal->asClass()->createInstance();
      // create the MemBuf we need; it refers the surface inside the carrier.
      MemBuf *colors = new MemBuf_4( (byte *) fmt->palette->colors, fmt->palette->ncolors, 0 );

      if ( carrier != 0 )
         colors->dependant( obj );

      objpal->setProperty( "colors", colors );
      objpal->setProperty( "ncolors", fmt->palette->ncolors );

      obj->setProperty( "palette", objpal );
   }

   return obj;
}

bool ObjectToPixelFormat( CoreObject *obj, ::SDL_PixelFormat *fmt )
{
   Item BitsPerPixel, BytesPerPixel;
   Item Rloss, Gloss, Bloss, Aloss;
   Item Rshift, Gshift, Bshift, Ashift;
   Item Rmask, Gmask, Bmask, Amask;
   Item colorkey, alpha;

   if ( !(
      obj->getProperty( "BitsPerPixel", BitsPerPixel ) &&
      obj->getProperty( "BytesPerPixel", BytesPerPixel ) &&
      obj->getProperty( "Rloss", Rloss ) &&
      obj->getProperty( "Gloss", Gloss ) &&
      obj->getProperty( "Bloss", Bloss ) &&
      obj->getProperty( "Aloss", Aloss ) &&

      obj->getProperty( "Rshift", Rshift ) &&
      obj->getProperty( "Gshift", Gshift ) &&
      obj->getProperty( "Bshift", Bshift ) &&
      obj->getProperty( "Ashift", Ashift ) &&

      obj->getProperty( "Rmask", Rmask ) &&
      obj->getProperty( "Gmask", Gmask ) &&
      obj->getProperty( "Bmask", Bmask ) &&
      obj->getProperty( "Amask", Amask ) &&
      obj->getProperty( "colorkey", colorkey ) &&
      obj->getProperty( "alpha", alpha ))
   )
      return false;

   fmt->BitsPerPixel = (Uint8) BitsPerPixel.forceInteger();
   fmt->BytesPerPixel = (Uint8) BytesPerPixel.forceInteger();
   fmt->Rloss = (Uint8) Rloss.forceInteger();
   fmt->Gloss = (Uint8) Gloss.forceInteger();
   fmt->Bloss = (Uint8) Bloss.forceInteger();
   fmt->Aloss = (Uint8) Aloss.forceInteger();

   fmt->Rshift = (Uint8) Rshift.forceInteger();
   fmt->Gshift = (Uint8) Gshift.forceInteger();
   fmt->Bshift = (Uint8) Bshift.forceInteger();
   fmt->Ashift = (Uint8) Ashift.forceInteger();

   fmt->Rmask = (Uint8) Rmask.forceInteger();
   fmt->Gmask = (Uint8) Gmask.forceInteger();
   fmt->Bmask = (Uint8) Bmask.forceInteger();
   fmt->Amask = (Uint8) Amask.forceInteger();
   fmt->colorkey = (Uint8) colorkey.forceInteger();
   fmt->alpha = (Uint8) alpha.forceInteger();

   return true;
}

/*#
   @class SDLVideoInfo
   @brief Encapsulate a video info.

   This class is returned from SDL.GetVideoInfo() and just stores some
   informations about the video.

   Consider its data as read only.

   @prop hw_available  Is it possible to create hardware surfaces?
   @prop wm_available  Is there a window manager available
   @prop blit_hw  Are hardware to hardware blits accelerated?
   @prop blit_hw_CC  Are hardware to hardware colorkey blits accelerated?
   @prop blit_hw_A  Are hardware to hardware alpha blits accelerated?
   @prop blit_sw  Are software to hardware blits accelerated?
   @prop blit_sw_CC  Are software to hardware colorkey blits accelerated?
   @prop blit_sw_A  Are software to hardware alpha blits accelerated?
   @prop blit_fill  Are color fills accelerated?
   @prop video_mem  Total amount of video memory in Kilobytes
   @prop vfmt  Pixel format of the video device stored in @a SDLPixelFormat instance
*/

CoreObject *MakeVideoInfo( VMachine *vm, const ::SDL_VideoInfo *info )
{

   Item *cls = vm->findWKI( "SDLVideoInfo" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();

   obj->setProperty( "hw_available", Item( info->hw_available ? true : false) );
   obj->setProperty( "wm_available", Item( info->wm_available ? true : false) );
   obj->setProperty( "blit_hw", Item( info->blit_hw ? true : false) );
   obj->setProperty( "blit_hw_CC", Item( info->blit_hw_CC ? true : false) );
   obj->setProperty( "blit_hw_A", Item( info->blit_hw_A ? true : false) );
   obj->setProperty( "blit_sw", Item( info->blit_sw ? true : false) );
   obj->setProperty( "blit_sw_CC", Item( info->blit_sw_CC ? true : false) );
   obj->setProperty( "blit_sw_A", Item( info->blit_sw_A ? true : false) );
   obj->setProperty( "blit_fill", Item( info->blit_fill ? true : false) );
   obj->setProperty( "video_mem", (int64) info->video_mem );
   // as video format info is stored in SDL, we shouldn't care about referencing it
   obj->setProperty( "vfmt",
       MakePixelFormatInst( vm, 0, info->vfmt ) );

   return obj;
}


}
}

/* end of sdl_mod.cpp */
