/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_mod.cpp

   The SDL binding support module - module specific extensions.
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
   The SDL binding support module - module specific extensions.
*/

#include <falcon/vm.h>
#include "sdl_mod.h"

extern "C" {
   #include <SDL.h>
}


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

SDLSurfaceCarrier::~SDLSurfaceCarrier()
{
   SDL_FreeSurface( m_surface );
}

bool SDLSurfaceCarrier::isReflective() const
{
   return true;
}

void SDLSurfaceCarrier::getProperty( VMachine *vm, const String &propName, Item &prop )
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
   else if ( propName == "clip_rect" )
   {
      prop = MakeRectInst( vm, m_surface->clip_rect );
   }
}

void SDLSurfaceCarrier::setProperty( VMachine *vm, const String &propName, Item &prop )
{
   // refuse to set anything
   vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Read Only Access" ) ) );
}


UserData *SDLSurfaceCarrier::clone() const
{
   return 0;
}

//==========================================
// Utilities
//

/*#
   @class SDLRect

   This class stores rectangular coordinates.
   Actually, this class is just a "contract" or "interface",
   as every function accepting an SDLRect will just accept any
   class providing the properties listed here.

   @prop x the X coordinate (left position).
   @prop y the Y coordinate (top position).
   @prop w width of the rectangle.
   @prop h height of the rectangle.
*/

bool RectToObject( const ::SDL_Rect &rect, CoreObject *obj )
{
   return obj->setProperty( "x", (int64) rect.x ) &&
          obj->setProperty( "y", (int64) rect.y ) &&
          obj->setProperty( "w", (int64) rect.w ) &&
          obj->setProperty( "h", (int64) rect.h );
}

bool ObjectToRect( CoreObject *obj, ::SDL_Rect &rect )
{
   Item *itm = obj->getProperty( "x" );
   if ( itm == 0 ) return false;
   rect.x = (Sint16) itm->forceInteger();

   *itm = obj->getProperty( "y" );
   if ( itm == 0 ) return false;
   rect.y = (Sint16) itm->forceInteger();

   *itm = obj->getProperty( "w" );
   if ( itm == 0 ) return false;
   rect.w = (Uint16) itm->forceInteger();

   *itm = obj->getProperty( "h" );
   if ( itm == 0 ) return false;
   rect.h = (Uint16) itm->forceInteger();
}

CoreObject *MakeRectInst( VMachine *vm, const ::SDL_Rect &rect )
{
   Item *cls = vm->findWKI( "SDLRect" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   RectToObject( rect, obj );
   vm->retval( obj );
}


}
}

/* end of sdl_mod.cpp */
