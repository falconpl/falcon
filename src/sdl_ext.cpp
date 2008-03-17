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

/*#
   @class SDL
   @brief SDL wrapper class

   This class acts as a module wrapper for SDL functions.
   The vast majority of the "SDL_" C namespace is provided by this module
   through access to SDL class.

   In example, to initialize the system through @b SDL_Init(), Falcon users
   should do the following:

   @code
      load SDL

      SDL.Init( \/* parameters *\/ )
   @endcode

   @note When using functions repeatedly, it is advisable to cache them to a local
   variable.
*/

//=================================================
// Initialization, quit and infos
//


/*#
   @method Init SDL
   @brief Initialize SDL system.
   @param flags SDL initialziation flags
   @throws SDLError on initialization failure

   This method initializes SDL. After initialization, it is necessary to call
   SDL.Quit() to clear the application state on exit; to avoid this need, it
   is possible to use the @a SDL.InitAuto Falcon extension.

   @note This "Init" method is not to be confused with Falcon language @b init statement.

   The following initialization flags can be provided; to specify more than one
   flag, use bitwise or (pipe "|" operator):

   - SDL.INIT_VIDEO: Initialize video subsystem.
   - SDL.INIT_AUDIO: Initialize audio subsystem.
   - SDL.INIT_TIMER: Prepare timers.
   - SDL.INIT_CDROM: Initialize CDROM interface.
   - SDL.INIT_JOYSTICK: Open joystick device.
   - SDL.INIT_EVERYTHING: ... initialize everything...
   - SDL.INIT_NOPARACHUTE: Do not intercept application critical signals (forced shutdowns).
*/
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

/*#
   @method InitAuto SDL
   @brief Initialize SDL system and provide automatic cleanup.
   @param flags SDL initialziation flags
   @return handle for SDL termination.
   @throws SDLError on initialization failure

   This method initializes SDL, and sets up an handle for SDL cleanup.
   The returned handle has a Quit() method that can be called autonomously
   by the application i.e. to return to text mode and perform final screen
   output. In case of abnormal termination (i.e. on exception raise), the
   handle calls its Quit() method automatically right before being collected
   at VM termination. If the handle is stored in a global variable of the main
   module, it will stay alive until the VM exits, and then it will be called
   to properly reset SDL status.

   @see SDL.Init
*/

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
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
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


/*#
   @method WasInit SDL
   @brief Detects if some SDL subsystem was initialized.
   @optparam flags SDL initialziation flags.
   @return a bitmask containing the initialized subsystems (among the ones requested).

   The parameter defautls to SDL_INIT_EVERYTHING if not provided.
*/

FALCON_FUNC sdl_WasInit( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init != 0 && ! i_init->isOrdinal() )
   {
       vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   Uint32 inMask = i_init == 0 ? SDL_INIT_EVERYTHING : (Uint32) i_init->forceInteger();
   Uint32 mask = ::SDL_WasInit( inMask );
   vm->retval( (int64) mask );
}

/*#
   @method Quit SDL
   @brief Turns off SDL system.

   This call shuts down SDL and resets system status. It is possible to turn off
   only some subsystems using the \a SDL.QuitSubSystem function.
*/

FALCON_FUNC sdl_Quit( ::Falcon::VMachine *vm )
{
   SDL_Quit();
}

/*#
   @method QuitSubSystem SDL
   @brief Shuts down a subset of SDL.
   @param subsys bitwise or'd init flags mask.

   This call shuts down one or more SDL subsystems.

   @see SDL.Init
*/
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

/*#
   @method SetVideoMode SDL
   @brief Changes the video mode and/or starts SDL context window.
   @param width - desired width
   @optparam bpp - byte per pixel in the desired modesired heightde (defaults to automatic)
   @optparam flags - flags to be eventually specified.
   @return a SDLScreen instance representing the SDL output device.
   @throws SDLError on initialization failure

   This function starts a graphic video mode and, on success, returns a @a SDLScreen
   class instance. This class is a Falcon object which encapsulate SDL_Surface structures,
   and has specific accessors and methods to handle the special SDL_Surface that is
   considered a "screen" by SDL.

   Flags can be one of the following value; also, some or'd combinations are possible.

   - SDL.SWSURFACE
   - SDL.HWSURFACE
   - SDL.ASYNCBLIT
   - SDL.ANYFORMAT
   - SDL.HWPALETTE
   - SDL.DOUBLEBUF
   - SDL.FULLSCREEN
   - SDL.OPENGL
   - SDL.OPENGLBLIT
   - SDL.RESIZABLE
   - SDL.NOFRAME

   For a complete explanation of the flags, please refer to SDL official documentation.

   @see SDLScreen
*/

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

/*#
   @method LoadBMP SDL
   @brief Loads an image from a BMP file.
   @param filename a filename to load.
   @return a @a SDLSurface instance containing the loaded image.
   @throws SDLError on load failure

   Loads an image from a BMP file and returns a new SDLSurface instance
   that can be manipulated through blit and similar functions.

   The filename is totally subject to SDL rules, as it is simply passed
   to SDL engine. No Falcon URI parsing is perfomred before invoking SDL;
   as such, it is advisable to use this function only in simple applications.

   @see SDLSurface
*/
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

/*#
   @method GetVideoSurface SDL
   @brief Retreives the current video surface.
   @return a @a SDLScreen instance representing the current video device.
   @throws SDLError on failure

   @see SDLScreen
*/
FALCON_FUNC sdl_GetVideoSurface( ::Falcon::VMachine *vm )
{
   ::SDL_Surface *surf = ::SDL_GetVideoSurface();
   if ( surf == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 5, __LINE__ )
         .desc( "SDL GetVideoSurface" )
         .extra( SDL_GetError() ) ) );
      return;
   }

   Item *cls = vm->findWKI( "SDLScreen" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance();
   obj->setUserData( new SDLSurfaceCarrier( surf ) );

   vm->retval( obj );
}


/*#
   @class SDLSurface
   @brief Encapsulates SDL_Surface structures and provides related services.

   This class is used to store SDL_Surface C structure that is widely used by SDL
   graphics engine. This class also provides the vast majority of the functions
   that operates on SDL surfaces as methods.

   Falcon provides a subclass called @a SDLScreen that provides also screen oriented
   functions; SDL do not differentiates between surfaces being used to handle image
   buffers and surfaces used as screen handles, but Falcon embedding provides a
   differentiation both for OOP cleanness and to minimize memory usage (i.e. to store method
   pointers unneeded by the vast majority of surfaces).
*/

/*#
   @method SaveBMP SDLSurface
   @brief Saves a BMP files to disk.
   @param filename the file where to store this BMP.
   @throws SDLError on failure.

   Save a memory image (or even a screenshot, if this surface is also a screen)
   to a disk BMP file.

   The filename is totally subject to SDL rules, as it is simply passed
   to SDL engine. No Falcon URI parsing is perfomred before invoking SDL;
   as such, it is advisable to use this function only in simple applications.

   @see SDL.LoadBMP
*/
FALCON_FUNC SDLSurface_SaveBMP( ::Falcon::VMachine *vm )
{
   Item *i_file = vm->param(0);
   if ( i_file == 0 || ! i_file->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   AutoCString fname( *i_file->asString() );

   CoreObject *self = vm->self().asObject();
   SDL_Surface *source = static_cast<SDLSurfaceCarrier*>( self->getUserData() )->m_surface;

   if ( ::SDL_SaveBMP( source, fname.c_str() ) < 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 5, __LINE__ )
         .desc( "SDL SaveBMP" )
         .extra( SDL_GetError() ) ) );
      return;
   }
   vm->retnil();
}

/*#
   @method BlitSurface SDLSurface
   @brief Copies a surface of part of it onto another surface.
   @param srcRect a @a SDLRect containing the source coordinates or nil.
   @param dest the destionation SDLSurface.
   @optparam dstRect a @a SDLRect containing destination coordinates or nil.
   @throws SDLError on copy failure.

   This functions copies a part of an image into another. The srcRect parameter determines
   which portion of the source image is copied; if nil, the whole image will be used.
   Only x and y coordinates from dstRect are used; if not provided, the image is copied
   starting at 0,0 (upper left corner).

*/

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

/*#
   @class SDLScreen
   @from SDLSurface
   @brief Screen oriented SDL surface.

   This class is a specialization of the SDLSurface class, providing methods
   meant to be used only on surfaces that should refer to SDL graphic devices.
*/

/*#
   @method UpdateRect SDLScreen
   @brief Refreshes an SDL screen surface or a part of it.
   @optparam xOrRect an X coordinate or a @a SDLRect containing refresh coordinates.
   @optparam y coordinate of the refresh rectangle.
   @optparam width width of the refresh rectangle.
   @optparam height height of the refresh rectangle.

   If no parameter is specified, the whole screen is refreshed. A refresh area can
   be specified either passing a single @a SDLRect instance as parameter or specifying
   the four coordinates as numbers.
*/

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

/*#
   @class SDLError
   @from Error
   @brief Class used to notify SDL exceptions.
*/

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
