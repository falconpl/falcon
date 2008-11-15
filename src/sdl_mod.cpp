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

#include <falcon/vm.h>
#include <falcon/membuf.h>
#include "sdl_mod.h"

extern "C" {
   #include <SDL.h>
}

/*# @beginmodule fsdl */

namespace Falcon {
namespace Ext {

//=======================================
// Quit carrier
//
QuitCarrier::~QuitCarrier()
{
   SDL_Quit();
}


//=======================================
// Surface reflection
//

SDLSurfaceCarrier_impl::~SDLSurfaceCarrier_impl()
{
   /*while( m_lockCount-- > 0 )
      SDL_UnlockSurface( m_surface );

   SDL_FreeSurface( m_surface );
*/
}


void SDLSurfaceCarrier_impl::getProperty( VMachine *vm, const String &propName, Item &prop )
{
   if ( propName == "w" )
   {
      prop = (int64) m_surface->w;
   }
   else if ( propName == "h" )
   {
      prop = (int64) m_surface->h;
   }
   else if ( propName == "flags" )
   {
      prop = (int64) m_surface->flags;
   }
   else if ( propName == "pitch" )
   {
      prop = (int64) m_surface->pitch;
   }
   else if ( propName == "bpp" )
   {
      prop = (int64) m_surface->format->BytesPerPixel;
   }
   else if ( propName == "clip_rect" )
   {
      if ( prop.isNil() )
         prop = MakeRectInst( vm, m_surface->clip_rect );
      else
         RectToObject( m_surface->clip_rect, prop.asObject() );
   }
   else if ( propName == "format" )
   {
      if ( prop.isNil() )
         prop = MakePixelFormatInst( vm, this );

   }
   else if ( propName == "pixels" )
   {
      if ( prop.isNil() )
      {
         MemBuf *mb;
         fassert( m_surface != 0 );

         switch( m_surface->format->BytesPerPixel )
         {
            case 1: mb = new MemBuf_1( vm, (byte*)m_surface->pixels, m_surface->h * m_surface->pitch, false );
            case 2: mb = new MemBuf_2( vm, (byte*)m_surface->pixels, m_surface->h * m_surface->pitch, false );
            case 3: mb = new MemBuf_3( vm, (byte*)m_surface->pixels, m_surface->h * m_surface->pitch, false );
            case 4: mb = new MemBuf_4( vm, (byte*)m_surface->pixels, m_surface->h * m_surface->pitch, false );
         }
         mb->dependant( vm->self().asObject() );
         prop = mb;
      }

   }
}

void SDLSurfaceCarrier_impl::setProperty( VMachine *vm, const String &propName, const Item &prop )
{
   // refuse to set anything
   vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Read Only Access" ) ) );
}


FalconData *SDLSurfaceCarrier_impl::clone() const
{
   return 0;
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

bool RectToObject( const ::SDL_Rect &rect, CoreObject *obj )
{
   return obj->setProperty( "x", (int64) rect.x ) &&
          obj->setProperty( "y", (int64) rect.y ) &&
          obj->setProperty( "w", (int64) rect.w ) &&
          obj->setProperty( "h", (int64) rect.h );
}

bool ObjectToRect( CoreObject *obj, ::SDL_Rect &rect )
{
   Item itm;
   if ( ! obj->getProperty( "x", itm ) ) return false;
   rect.x = (Sint16) itm.forceInteger();

   if ( ! obj->getProperty( "y", itm ) ) return false;
   rect.y = (Sint16) itm.forceInteger();

   if ( ! obj->getProperty( "w", itm ) ) return false;
   rect.w = (Sint16) itm.forceInteger();

   if ( ! obj->getProperty( "h", itm ) ) return false;
   rect.h = (Sint16) itm.forceInteger();
   return true;
}

CoreObject *MakeRectInst( VMachine *vm, const ::SDL_Rect &rect )
{
   Item *cls = vm->findWKI( "SDLRect" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   RectToObject( rect, obj );
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
      MemBuf *colors = new MemBuf_4( vm,
         (byte *) fmt->palette->colors, fmt->palette->ncolors, false );

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

   fmt->BitsPerPixel = BitsPerPixel.forceInteger();
   fmt->BytesPerPixel = BytesPerPixel.forceInteger();
   fmt->Rloss = Rloss.forceInteger();
   fmt->Gloss = Gloss.forceInteger();
   fmt->Bloss = Bloss.forceInteger();
   fmt->Aloss = Aloss.forceInteger();

   fmt->Rshift = Rshift.forceInteger();
   fmt->Gshift = Gshift.forceInteger();
   fmt->Bshift = Bshift.forceInteger();
   fmt->Ashift = Ashift.forceInteger();

   fmt->Rmask = Rmask.forceInteger();
   fmt->Gmask = Gmask.forceInteger();
   fmt->Bmask = Bmask.forceInteger();
   fmt->Amask = Amask.forceInteger();
   fmt->colorkey = colorkey.forceInteger();
   fmt->alpha = alpha.forceInteger();

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
