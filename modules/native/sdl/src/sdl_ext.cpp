/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_ext.cpp

   The SDL binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Mar 2008 19:37:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

#include "SDL.h"

/*# @beginmodule sdl */

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
      load sdl

      SDL.Init( /\* parameters *\/ )
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
   @raise SDLError on initialization failure

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

   As Falcon strings and character manipulation is mainly unicode, this init method automatically
   enables SDL unicode in event generation. If this is not desired, it is possible to call
   @a SDL.EnableUNICODE to disable it after init.
*/
FALCON_FUNC sdl_Init( ::Falcon::VMachine *vm )
{
   Item *i_init = vm->param(0);

   if ( i_init == 0 || ! i_init->isOrdinal() )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) ;
      return;
   }

   int retval = ::SDL_Init( (Uint32) i_init->forceInteger() );
   if ( retval < 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Error" )
         .extra( SDL_GetError() ) ) ;
   }

   // Falcon is all about unicode.
   SDL_EnableUNICODE( 1 );
}

/*#
   @method InitAuto SDL
   @brief Initialize SDL system and provide automatic cleanup.
   @param flags SDL initialization flags
   @return handle for SDL termination.
   @raise SDLError on initialization failure

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
       throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) ;
      return;
   }

   int retval = ::SDL_Init( (Uint32) i_init->forceInteger() );
   if ( retval < 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Init error" )
         .extra( SDL_GetError() ) ) ;
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
       throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) ;
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
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) ;
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
   @raise SDLError on error.

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
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 6, __LINE__ )
         .desc( "SDL Video Info error" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   vm->retval( MakeVideoInfo( vm, vi ) );
}

/*#
   @method VideoDriverName SDL
   @brief Returns the name of the used video driver.
   @return A simple description of the video driver being used.
   @raise SDLError if the system is not initialized.
*/
FALCON_FUNC sdl_VideoDriverName( ::Falcon::VMachine *vm )
{
   char name[1024];

   if ( SDL_VideoDriverName( name, 1023) == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE, __LINE__ )
         .desc( "SDL Init error" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   vm->retval( new CoreString( name ) );
}

/*#
   @method ListModes SDL
   @brief Returns a list of possible modes.
   @return An array of x,y pairs containing the sizes of
      the available modes, nil or -1.
   @optparam format An SDLPixelFormat structure to filter modes or nil.
   @optparam flags Initialization flags that must be supported by
             returned mode.
   @raise SDLError if the system is not initialized.

   The function mimics the workings of SDL_ListModes, returning an array
   of pairs (2 element arrays) with x, y items, if a set of mode was found,
   nil if no mode is available and -1 if SDL says "all the modes" are
   available.

   If passing nil as the desired pixel format, then the default screen
   pixel format will be used.
*/

FALCON_FUNC sdl_ListModes( ::Falcon::VMachine *vm )
{
   if ( SDL_WasInit(0) == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE+1, __LINE__ ).
         desc( "SDL not initialized" ) );
   }
   
   Item *i_format = vm->param(0);
   Item *i_flags = vm->param(1);
   

   if ( (i_format != 0 &&
        (! i_format->isNil() &&
        ( ! i_format->isObject() || ! i_format->asObject()->derivedFrom( "SDLPixelFormat") ))) ||
      ( i_flags != 0 && ! i_flags->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[SDLPixelFormat, N]" ) );
      return;
   }

   SDL_PixelFormat fmt, *pFmt;
   if( i_format != 0 && ! i_format->isNil() )
   {
      pFmt = &fmt;
      ObjectToPixelFormat( i_format->asObject(), pFmt );
   }
   else 
   {
      const SDL_VideoInfo* myPointer = SDL_GetVideoInfo();
      pFmt = myPointer->vfmt;
   }

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
      CoreArray *array = new CoreArray;
      int pos = 0;
      while( list[pos] != 0 )
      {
         CoreArray *res = new CoreArray( 2 );
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
        ( i_bpp != 0 && ! i_bpp->isOrdinal() ) ||
        ( i_flags != 0 && ! i_flags->isOrdinal() )
      )
   {
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,[N,N]" ) ) ;
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
   @raise SDLError on initialization failure

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
        ( i_bpp != 0 && ! i_bpp->isOrdinal() ) ||
        ( i_flags != 0 && ! i_flags->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,[N,N]" ) ) ;
      return;
   }

   int width = (int) i_width->forceInteger();
   int height = (int) i_height->forceInteger();
   int bpp = i_bpp == 0 ? 0 : (int) i_bpp->asInteger();
   int flags = i_flags == 0 ? 0 : (int) i_flags->asInteger();

   SDL_Surface *screen = ::SDL_SetVideoMode( width, height, bpp, flags );

   if ( screen == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 2, __LINE__ )
         .desc( "SDL Set video mode error" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   Item *cls = vm->findWKI( "SDLScreen" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance( screen );

   // SDL free must NOT be called on screen surfaces.
   // For this reason, we leave it +1 referenced, so it will be never deleted
   //SDL_FreeSurface( screen );
   screen->refcount++;

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
   @raise SDLError on load failure

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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) ;
      return;
   }

   AutoCString fname( *i_file->asString() );

   ::SDL_Surface *surf = ::SDL_LoadBMP( fname.c_str() );
   if ( surf == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 3, __LINE__ )
         .desc( "SDL LoadBMP" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   Item *cls = vm->findWKI( "SDLSurface" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance( surf );
   SDL_FreeSurface( surf );
   vm->retval( obj );
}

/*#
   @method GetVideoSurface SDL
   @brief Retreives the current video surface.
   @return a @a SDLScreen instance representing the current video device.
   @raise SDLError on failure

   @see SDLScreen
*/
FALCON_FUNC sdl_GetVideoSurface( ::Falcon::VMachine *vm )
{
   ::SDL_Surface *surf = ::SDL_GetVideoSurface();
   if ( surf == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 5, __LINE__ )
         .desc( "SDL GetVideoSurface" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   Item *cls = vm->findWKI( "SDLScreen" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance( surf );

   // SDL free must NOT be called on screen surfaces.
   // For this reason, we leave it +1 referenced, so it will be never deleted
   //SDL_FreeSurface( surf );
   surf->refcount++;

   vm->retval( obj );
}

/*#
   @method SetGamma SDL
   @brief Sets the gamma function values for the active SDL output.
   @param red Red gamma correction value.
   @param green Green gamma correction value.
   @param blue Blue gamma correction value.
   @raise SDLError if the hardware doesn't support gamma
*/
FALCON_FUNC sdl_SetGamma ( ::Falcon::VMachine *vm )
{
   Item *i_red;
   Item *i_green;
   Item *i_blue;

   if( vm->paramCount() < 3 ||
      ! ( i_red = vm->param(0) )->isOrdinal() ||
      ! ( i_green = vm->param(1) )->isOrdinal() ||
      ! ( i_blue = vm->param(2) )->isOrdinal()
   )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,N" ) ) ;
      return;
   }

   int res = ::SDL_SetGamma( i_red->forceNumeric(), i_green->forceNumeric(), i_blue->forceNumeric() );

   if ( res == -1 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 8, __LINE__ )
         .desc( "SDL Set Gamma" )
         .extra( SDL_GetError() ) ) ;
      return;
   }
}

/*#
   @method GetGammaRamp SDL
   @brief Get Gamma ramps for this hardware.
   @optparam aRet An array that will contain the gamma memory buffers on exit.
   @return An array containing the three MemBufs
   @raise SDLError if the hardware doesn't support gamma.

   This functions returns three membuf that maps directly the gamma correction
   table for the red, green and blue value.

   Each membuf is a 2 bytes memory vector of 256 binary values.
*/

FALCON_FUNC sdl_GetGammaRamp ( ::Falcon::VMachine *vm )
{
   Item *i_arr = vm->param(0);

   if( i_arr != 0 && ! i_arr->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,N" ) ) ;
      return;
   }

   CoreArray *arr = i_arr == 0 ? new CoreArray( 3 ) : i_arr->asArray();
   arr->length(0);

   MemBuf *red_buf = new MemBuf_2( 256 );
   MemBuf *green_buf = new MemBuf_2( 256 );
   MemBuf *blue_buf = new MemBuf_2( 256 );

   int res = ::SDL_GetGammaRamp(
      (Uint16 *) red_buf->data(),
      (Uint16 *) green_buf->data(),
      (Uint16 *) blue_buf->data()
      );

   if ( res == -1 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 8, __LINE__ )
         .desc( "SDL Get Gamma Ramp" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   arr->append( red_buf );
   arr->append( green_buf );
   arr->append( blue_buf );

   vm->retval( arr );
}

/*#
   @method SetGammaRamp SDL
   @brief Set Gamma ramps for this hardware.
   @param redbuf A 2 bytes 256 elements memory buffer for the red channel, or nil.
   @param greenbuf A 2 bytes 256 elements memory buffer for the blue channel, or nil.
   @param bluebuf A 2 bytes 256 elements memory buffer for the green channel, or nil.
   @raise SDLError if the hardware doesn't support gamma.

   Each membuf is a 2 bytes memory vector of 256 binary values. If one of the channels
   needs not to be changed, nil can be placed instead.

*/
FALCON_FUNC sdl_SetGammaRamp ( ::Falcon::VMachine *vm )
{
   Item *i_redmb = vm->param(0);
   Item *i_greenmb = vm->param(1);
   Item *i_bluemb = vm->param(2);

   if( i_redmb == 0 || ( ! i_redmb->isMemBuf() && ! i_redmb->isNil() ) ||
       i_greenmb == 0 || ( ! i_greenmb->isMemBuf() && ! i_greenmb->isNil() ) ||
       i_bluemb == 0 || ( ! i_bluemb->isMemBuf() && ! i_bluemb->isNil() )
   )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "M|Nil,M|nil,M|Nil" ) ) ;
      return;
   }

   Uint16 *red=0, *green=0, *blue=0;
   bool valid = true;
   // non-nil membuf must be 2 bytes wide 256 elements
   if( i_redmb->isNil() )
   {
      red = 0;
   }
   else {
      MemBuf *mb = i_redmb->asMemBuf();
      if( mb->length() != 256 || mb->wordSize() != 2 )
         valid = false;
      else
         red = (Uint16 *) mb->data();
   }

   if( i_greenmb->isNil() )
   {
      green = 0;
   }
   else {
      MemBuf *mb = i_greenmb->asMemBuf();
      if( mb->length() != 256 || mb->wordSize() != 2 )
         valid = false;
      else
         green = (Uint16 *) mb->data();
   }

   if( i_bluemb->isNil() )
   {
      blue = 0;
   }
   else {
      MemBuf *mb = i_bluemb->asMemBuf();
      if( mb->length() != 256 || mb->wordSize() != 2 )
         valid = false;
      else
         blue = (Uint16 *) mb->data();
   }

   if( ! valid )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ) ) ;
      return;
   }

   int res = ::SDL_SetGammaRamp( red, green, blue );

   if ( res == -1 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 8, __LINE__ )
         .desc( "SDL Get Gamma Ramp" )
         .extra( SDL_GetError() ) ) ;
      return;
   }
}


static void sdl_CreateRGBSurface_internal ( ::Falcon::VMachine *vm, MemBuf *mb, int flags )
{
   Item *i_width = 0;
   Item *i_height = 0;
   Item *i_depth = 0;

   Uint32 redMask;
   Uint32 greenMask;
   Uint32 blueMask;
   Uint32 alphaMask;
   int depth;

   int pcount = vm->paramCount();
   bool bValid = true;
   if ( ( pcount == 4 ) || (pcount == 8 ) )
   {
      if(
         ! (i_width = vm->param(1) )->isOrdinal() ||
         ! (i_height = vm->param(2) )->isOrdinal() ||
         ! (i_depth = vm->param(3) )->isOrdinal()
      )
      {
         bValid = false;
      }
      else {
         // do we have to calculate the bitmaps?
         depth = i_depth->forceInteger();

         // check on depth
         if ( mb != 0 && (
              depth/8 != mb->wordSize() ||
              mb->length() != (i_width->forceInteger() * i_height->forceInteger() ) )
            )
         {
            throw new ParamError( ErrorParam( e_param_range, __LINE__ ).
               extra( "Membuf not matching sizes" ) ) ;
            return;
         }
         else if( pcount == 8  )
         {
            Item *i_red, *i_green, *i_blue, *i_alpha;

            if(
               ! (i_red = vm->param(4) )->isInteger() ||
               ! (i_green = vm->param(5) )->isInteger() ||
               ! (i_blue = vm->param(6) )->isInteger() ||
               ! (i_alpha = vm->param(7) )->isInteger()
               )
            {
               bValid = false;
            }
            else {
               // get our data.
               redMask = (Uint32) i_red->asInteger();
               greenMask = (Uint32) i_green->asInteger();
               blueMask = (Uint32) i_blue->asInteger();
               alphaMask = (Uint32) i_alpha->asInteger();
            }
         }
         else {
            // we have to calculate the value on our own
            uint32 base = 0;
            uint32 colorSize = depth/4;
            for ( uint32 i = 0; i < colorSize; i ++ )
            {
               base |= 1 << i;
            }

            #if SDL_BYTEORDER == SDL_BIG_ENDIAN
               redMask = base << (depth - colorSize);
               blueMask = base << (depth - colorSize *2);
               greenMask = base << (depth - colorSize *3);
               alphaMask = base;
            #else
               alphaMask = base << (depth - colorSize);
               blueMask = base << (depth - colorSize *2);
               greenMask = base << (depth - colorSize *3);
               redMask = base;
            #endif
         }
      }
   }
   else {
      bValid = false;
   }

   // invalid parameters?
   if( ! bValid )
   {
      const char *extra = mb == 0 ? "I,N,N,N,[I,I,I,I]" : "M,N,N,N,[I,I,I,I]";
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( extra ) ) ;
      return;
   }

   SDL_Surface *surf;
   if ( mb == 0 )
      surf = SDL_CreateRGBSurface(
            flags,
            i_width->forceInteger(), i_height->forceInteger(), depth,
            redMask, greenMask, blueMask, alphaMask
         );
   else
      surf = SDL_CreateRGBSurfaceFrom( mb->data(),
            i_width->forceInteger(), i_height->forceInteger(),
            depth, mb->wordSize(),
            redMask, greenMask, blueMask, alphaMask
         );

   // Success?
   if ( surf == 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 9, __LINE__ )
         .desc( "SDL Create RGB Surface error" )
         .extra( SDL_GetError() ) ) ;
      return;
   }

   Item *cls = vm->findWKI( "SDLSurface" );
   fassert( cls != 0 );
   CoreObject *obj = cls->asClass()->createInstance( surf );
   SDL_FreeSurface( surf );
   // if we have a membuf, store it in the right property so it stays alive
   if ( mb != 0 )
   {
      dyncast<SDLSurfaceCarrier_impl* >(obj)->setPixelCache( mb );
   }

   vm->retval( obj );
}

/*#
   @method CreateRGBSurface SDL
   @brief Creates a paintable surface
   @param flags Creation flags.
   @param width Width of the created buffer.
   @param height Height of the created buffer.
   @param depth Bit per pixel depth of the image - can be 8, 16, 24 or 32.
   @optparam rMask Red bitmask - defaults to low address bits.
   @optparam gMask Green bitmask - defaults to second low address bits.
   @optparam bMask Blue bitmask - defaults to third low address bits.
   @optparam aMask Alpha bitmask - defaults to hihest address bits.
   @return The newly created surface.
   @raise SDLError on creation error.

   The function can be called either with 4 or 8 parameters. 8 bits per pixel modes
   don't require bit masks, and they are ignored if provided; other modes require
   a bitmask. If the values are not provided, this function calculates them using
   a R,G,B,A evenly spaced bitmap, which will place the bits stored in the lowest
   address in the red space. On big endian machines, red will be placed in the most
   signficant bits, on little endian it will be place on the least significant bits.

   You should provide your own bitmap values if you don't want alpha surfaces.

   The flags can be a combination of the following:

   - SDL.SWSURFACE - SDL will create the surface in system memory.
                    This improves the performance of pixel level access,
                    however you may not be able to take advantage of some
                    types of hardware blitting.
   - SDL.HWSURFACE - SDL will attempt to create the surface in video memory.
                    This will allow SDL to take advantage of Video->Video blits
                    (which are often accelerated).
   - SDL.SRCCOLORKEY - This flag turns on colourkeying for blits from this surface.
                     If SDL_HWSURFACE is also specified and colourkeyed blits are
                     hardware-accelerated, then SDL will attempt to place the surface
                     in video memory. Use SDL_SetColorKey to set or clear this flag
                     after surface creation.
   - SDL.SRCALPHA - This flag turns on alpha-blending for blits from this surface. If
                    SDL_HWSURFACE is also specified and alpha-blending blits are
                    hardware-accelerated, then the surface will be placed in video memory
                    if possible. Use SDL_SetAlpha to set or clear this flag after
                    surface creation.
*/
FALCON_FUNC sdl_CreateRGBSurface ( ::Falcon::VMachine *vm )
{
   Item *i_flags = vm->param(0);
   if( ! i_flags->isInteger() )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).
         extra( "I,N,N,N,[I,I,I,I]") ) ;
   }
   else
   {
      sdl_CreateRGBSurface_internal( vm, 0 , i_flags->asInteger() );
   }
}

/*#
   @method CreateRGBSurfaceFrom SDL
   @brief Creates a paintable surface using existing data.
   @param pixels Original pixels in a membuf.
   @param width Width of the created buffer
   @param height Height of the created buffer.
   @param depth Bit per pixel depth of the image -- must match pixel membuf word length.
   @optparam rMask Red bitmask - defaults to low address bits.
   @optparam gMask Green bitmask - defaults to second low address bits.
   @optparam bMask Blue bitmask - defaults to third low address bits.
   @optparam aMask Alpha bitmask - defaults to hihest address bits.
   @return The newly created surface.
   @raise SDLError on creation error.

   @see SDL.CreateRGBSurface
*/
FALCON_FUNC sdl_CreateRGBSurfaceFrom ( ::Falcon::VMachine *vm )
{
   Item *i_pixels = vm->param(0);
   if( ! i_pixels->isMemBuf() )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).
         extra( "M,N,N,N,[I,I,I,I]") ) ;
   }
   else
   {
      sdl_CreateRGBSurface_internal( vm, i_pixels->asMemBuf(), 0 );
   }
}

//==================================================================
// SDLRect class
//

/*#
   @class SDLRect
   @brief Storage for rectangular coordinates.

   This class stores rectangular coordinates.
   Actually, this class is just a "contract" or "interface",
   as every function accepting an SDLRect will just accept any
   class providing the properties listed here.

   @prop x the X coordinate (left position).
   @prop y the Y coordinate (top position).
   @prop w width of the rectangle.
   @prop h height of the rectangle.
*/

/*#
   @init SDLRect
   @brief Initializes the rectangle.
   @optparam x X cooordinate of this rectangle
   @optparam y Y cooordinate of this rectangle
   @optparam w width of this rectangle
   @optparam h height of this rectangle

   Fills the rectangle with initial values.
*/

FALCON_FUNC SDLRect_init( ::Falcon::VMachine *vm )
{
   Item *i_x = vm->param(0);
   Item *i_y = vm->param(1);
   Item *i_width = vm->param(2);
   Item *i_height = vm->param(3);

   if (
        ( i_x != 0 && ! i_x->isOrdinal() ) ||
        ( i_y != 0 && ! i_y->isOrdinal() ) ||
        ( i_width != 0 && ! i_width->isOrdinal() ) ||
        ( i_height != 0 && ! i_height->isOrdinal() )
      )
   {
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N,N,N,N]" ) ) ;
      return;
   }

   SDL_Rect* r = (SDL_Rect*) memAlloc( sizeof( SDL_Rect ) );
   vm->self().asObject()->setUserData(r);
   r->x = i_x == 0 ? 0 : i_x->forceInteger();
   r->y = i_y == 0 ? 0 : i_y->forceInteger();
   r->w = i_width == 0 ? 0 : i_width->forceInteger();
   r->h = i_height == 0 ? 0 : i_height->forceInteger();
}

//==================================================================
// COLOR class
//

/*#
   @init SDLColor
   @brief Set initial values for this color.
   @param r Red value
   @param g Green value
   @param b Blue value

*/
FALCON_FUNC SDLColor_init( VMachine *vm )
{
   Item *i_r, *i_g, *i_b;

   if ( vm->paramCount() < 3 ||
       ! ( i_r = vm->param(0) )->isOrdinal() ||
       ! ( i_g = vm->param(1) )->isOrdinal() ||
       ! ( i_b = vm->param(2) )->isOrdinal()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N,N" ) ) ;
      return;
   }

   SDL_Color* c = (SDL_Color*) memAlloc( sizeof( SDL_Color ) );
   vm->self().asObject()->setUserData( c );
   c->r = i_r->forceInteger();
   c->g = i_g->forceInteger();
   c->b = i_b->forceInteger();
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
      einst->setUserData( new SDLError );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of sdl_ext.cpp */
