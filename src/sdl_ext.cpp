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
#include <falcon/membuf.h>

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

/*#
   @method IsBigEndian SDL
   @brief Returns true if the host system is big endian.
   @return true if the host system is big endian.
*/
FALCON_FUNC sdl_IsBigEndian( ::Falcon::VMachine *vm )
{
   vm->retval( SDL_BYTEORDER == SDL_BIG_ENDIAN );
}

/*#
   @method GetVideoInfo SDL
   @brief Returns details about underlying video system.
   @return An instance of a @a SDLVideoInfo
   @throws SDLError on error.

   This function returns a read-only SDLVideoInfo instance containing informations about
   the video hardware.

   If this is called before SDL_SetVideoMode, the vfmt member of the returned
   structure will contain the pixel format of the "best" video mode.
*/
FALCON_FUNC sdl_GetVideoInfo( ::Falcon::VMachine *vm )
{
   const SDL_VideoInfo *vi = SDL_GetVideoInfo();
   if ( vi == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 6, __LINE__ )
         .desc( "SDL Video Info error" )
         .extra( SDL_GetError() ) ) );
      return;
   }

   vm->retval( MakeVideoInfo( vm, vi ) );
}

/*#
   @method VideoDriverName SDL
   @brief Returns the name of the used video driver.
   @return A simple description of the video driver being used.
   @throws SDLError if the system is not initialized.
*/
FALCON_FUNC sdl_VideoDriverName( ::Falcon::VMachine *vm )
{
   char name[1024];

   if ( SDL_VideoDriverName( name, 1023) == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Init error" )
         .extra( SDL_GetError() ) ) );
      return;
   }

   vm->retval( new GarbageString( vm, name ) );
}

/*#
   @method ListModes SDL
   @brief Returns a list of possible modes.
   @return An array of x,y pairs containing the sizes of
      the available modes, nil or -1.
   @optparam format An SDLPixelFormat structure to filter modes or nil.
   @optparam flags Initialization flags that must be supported by
             returned mode.
   @throws SDLError if the system is not initialized.

   The function mimics the workings of SDL_ListModes, returning an array
   of pairs (2 element arrays) with x, y items, if a set of mode was found,
   nil if no mode is available and -1 if SDL says "all the modes" are
   available.

   If passing nil as the desired pixel format, then the default screen
   pixel format will be used.
*/

FALCON_FUNC sdl_ListModes( ::Falcon::VMachine *vm )
{
   Item *i_format = vm->param(0);
   Item *i_flags = vm->param(1);

   if ( (i_format != 0 &&
        (! i_format->isNil() &&
        ( ! i_format->isObject() || ! i_format->asObject()->derivedFrom( "SDLPixelFormat") ))) ||
      ( i_flags != 0 && ! i_flags->isOrdinal() ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[SDLPixelFormat, N]" ) ) );
      return;
   }

   SDL_PixelFormat fmt, *pFmt;
   if( i_format != 0 && ! i_format->isNil() )
   {
      pFmt = &fmt;
      ObjectToPixelFormat( i_format->asObject(), pFmt );
   }
   else
      pFmt = 0;

   Uint32 flags = i_flags == 0 ? 0 : (Uint32) i_flags->forceInteger();

   SDL_Rect **list = SDL_ListModes( pFmt, flags );
   if ( list == 0 )
   {
      vm->retnil();
   }
   else if ( list == (SDL_Rect**) - 1)
   {
      vm->retval( (int64) -1 );
   }
   else {
      CoreArray *array = new CoreArray( vm );
      int pos = 0;
      while( list[pos] != 0 )
      {
         CoreArray *res = new CoreArray( vm, 2 );
         array->append( res );
         res->append( (int64) list[pos]->w );
         res->append( (int64) list[pos]->h );
         ++pos;
      }
      vm->retval( array );
   }
}

//==================================================================
// Generic video mode
//

/*#
   @method VideoModeOK SDL
   @brief Verifies if a given video mode is ok
   @param width - desired width.
   @param height - desired height.
   @optparam bpp - byte per pixel in the desired modesired heightde (defaults to automatic).
   @optparam flags - flags to be eventually specified.
   @return 0 if the mode isn't available, or a bit per pixel value for the given mode.

   The function can be used to checked if a video mode can be used prior to starting it.
   If the mode is available, a preferred bit per pixel value is returned, otherwise
   the function returns zero.
*/

FALCON_FUNC sdl_VideoModeOK( ::Falcon::VMachine *vm )
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

   int dbpp = ::SDL_VideoModeOK( width, height, bpp, flags );

   vm->retval( (int64) dbpp );
}

/*#
   @method SetVideoMode SDL
   @brief Changes the video mode and/or starts SDL context window.
   @param width - desired width.
   @param height - desired height.
   @optparam bpp - byte per pixel in the desired modesired heightde (defaults to automatic).
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
   obj->setUserData( new SDLSurfaceCarrier( vm, screen ) );

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
   obj->setUserData( new SDLSurfaceCarrier( vm, surf ) );

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
   obj->setUserData( new SDLSurfaceCarrier( vm, surf ) );

   vm->retval( obj );
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
