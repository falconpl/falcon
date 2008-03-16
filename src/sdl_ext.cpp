/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_ext.cpp

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

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>

#include "sdl_ext.h"
#include "sdl_mod.h"

#include <SDL.h>

namespace Falcon {
namespace Ext {


//=================================================
// Initialization, quit and infos
//

FALCON_FUNC sdl_Init( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   int retval = ::SDL_Init( (Uint32) i_init->forceInteger() );
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Error" )
         .extra( SDL_GetError() ) ) );
   }
}


FALCON_FUNC sdl_InitAuto( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   int retval = ::SDL_Init( (Uint32) i_init->forceInteger() );
   if ( retval < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 1, __LINE__ )
         .desc( "SDL Init error" )
         .extra( SDL_GetError() ) ) );
      return;
   }

   // also create an object for auto quit.
   Item *c_auto = vm->findWKI( "_SDL_AutoQuit" );
   CoreObject *obj = c_auto->asClass()->createInstance();
   obj->setUserData( new QuitCarrier );
   vm->retval( obj );
}


FALCON_FUNC sdl_WasInit( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   Uint32 mask = ::SDL_WasInit( (Uint32) i_init->forceInteger() );
   vm->retval( (int64) mask );
}


FALCON_FUNC sdl_Quit( ::Falcon::VMachine *vm )
{
   SDL_Quit();
}

FALCON_FUNC sdl_QuitSubSystem( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   ::SDL_QuitSubSystem( (Uint32) i_init->forceInteger() );
}

//==================================================================
// Generic video mode
//
FALCON_FUNC sdl_SetVideoMode( ::Falcon::VMachine *vm )
{
   Item *i_width = vm->param(0);
   Item *i_height = vm->param(1);
   Item *i_bpp = vm->param(2);
   Item *i_flags = vm->param(3);

   if ( ( i_width == 0 || ! i_width->isOrdinal() ) ||
        ( i_height == 0 || ! i_height->isOrdinal() ) ||
        ( i_bpp != 0 && ! i_height->isOrdinal() ) ||
        ( i_flags != 0 && ! i_flags->isOrdinal() )
      )
   {
       vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,[N,N]" ) ) );
      return;
   }

   int width = (int) i_width->forceInteger();
   int height = (int) i_height->forceInteger();
   int bpp = i_bpp == 0 ? 0 : (int) i_bpp->asInteger();
   int flags = i_flags == 0 ? 0 : (int) i_flags->asInteger();

   SDL_Surface *screen = ::SDL_SetVideoMode( width, height, bpp, flags );

   if ( screen == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 2, __LINE__ )
         .desc( "SDL Set video mode error" )
         .extra( SDL_GetError() ) ) );
      return;
   }

   Item *cls = vm->findWKI( "SDLScreen" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   obj->setUserData( new SDLSurfaceCarrier( screen ) );

   vm->retval( obj );
}

//==================================================================
// Surface video mode
//

FALCON_FUNC sdl_LoadBMP( ::Falcon::VMachine *vm )
{
   Item *i_file = vm->param(0);
   if ( i_file == 0 || ! i_file->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   AutoCString fname( *i_file->asString() );

   ::SDL_Surface *surf = ::SDL_LoadBMP( fname.c_str() );
   if ( surf == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 3, __LINE__ )
         .desc( "SDL LoadBMP" )
         .extra( SDL_GetError() ) ) );
      return;
   }

   Item *cls = vm->findWKI( "SDLSurface" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   obj->setUserData( new SDLSurfaceCarrier( surf ) );

   vm->retval( obj );
}


FALCON_FUNC SDLSurface_BlitSurface( ::Falcon::VMachine *vm )
{
   // rects can be nil, in which case we must set them to null
   Item *i_srcrect = vm->param(0);
   Item *i_dest = vm->param(1);
   Item *i_dstrect = vm->param(2);

   bool paramOk = true;
   SDL_Rect srcRect, dstRect, *pSrcRect, *pDstRect;

   if ( ( i_srcrect == 0 || ( ! i_srcrect->isNil() && ! i_srcrect->isObject() )) ||
        ( i_dest == 0 || ! i_dest->isObject() || ! i_dest->asObject()->derivedFrom( "SDLSurface" )) ||
        ( i_dstrect != 0 && ! i_dstrect->isNil() && ! i_dstrect->isObject() )
      )
   {
      paramOk = false;
   }
   else
   {
      // are rects exposing the right interface?
      if( i_srcrect != 0 && i_srcrect->isObject() )
      {
         if ( ! ObjectToRect( i_srcrect->asObject(), srcRect ) )
            paramOk = false;
         else
            pSrcRect = &srcRect;
      }
      else
         pSrcRect = 0;

      if ( paramOk )
      {
         if( i_dstrect != 0 && i_dstrect->isObject() )
         {
            if ( ! ObjectToRect( i_dstrect->asObject(), dstRect ) )
               paramOk = false;
            else
               pDstRect = &dstRect;
         }
         else
            pDstRect = 0;
      }
   }


   // are our parameter ok?
   if ( ! paramOk )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "SDLRect|Nil, SDLSurface [, SDLRect|Nil]" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   SDL_Surface *source = static_cast<SDLSurfaceCarrier*>( self->getUserData() )->m_surface;
   SDL_Surface *dest = static_cast<SDLSurfaceCarrier*>( i_dest->asObject()->getUserData() )->m_surface;

   int res = ::SDL_BlitSurface( source, pSrcRect, dest, pDstRect );
   if( res < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 4, __LINE__ )
         .desc( "SDL BlitSurface" )
         .extra( SDL_GetError() ) ) );
      return;
   }

}


FALCON_FUNC SDLScreen_UpdateRect( ::Falcon::VMachine *vm )
{
   // accept a rect as first parameter, or even no parameter.
   CoreObject *self = vm->self().asObject();
   SDL_Surface *screen = static_cast<SDLSurfaceCarrier*>( self->getUserData() )->m_surface;

   if( vm->paramCount() == 0 )
   {
      ::SDL_UpdateRect( screen, 0, 0, 0, 0);
   }
   else if ( vm->paramCount() == 1 )
   {
      Item *i_rect = vm->param(1);
      SDL_Rect r;

      if( ! i_rect->isObject() || ! ObjectToRect( i_rect->asObject(), r ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "SDLRect|N,[N,N,N]" ) ) );
         return;
      }

      ::SDL_UpdateRect( screen, r.x, r.y, r.w, r.w );
   }
   else
   {
      Item *i_x = vm->param(0);
      Item *i_y = vm->param(1);
      Item *i_w = vm->param(2);
      Item *i_h = vm->param(3);

      if( i_x == 0  || ! i_x->isOrdinal() ||
          i_y == 0  || ! i_y->isOrdinal() ||
          i_w == 0  || ! i_w->isOrdinal() ||
          i_h == 0  || ! i_h->isOrdinal()
        )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "SDLRect|N,[N,N,N]" ) ) );
         return;
      }

      ::SDL_UpdateRect( screen, i_x->forceInteger(), i_y->forceInteger(),
                                i_w->forceInteger(), i_h->forceInteger() );
   }
}


//==================================================================
// ERROR class
//

FALCON_FUNC  SDLError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new SDLError ) );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of sdl_ext.cpp */
