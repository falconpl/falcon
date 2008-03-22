/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_wm_ext.cpp

   The SDL binding support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 22 Mar 2008 16:02:18 +0100
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
   @method WM_SetCaption SDL
   @brief Set Caption for SDL window and for window icon.
   @param caption String containing the window caption.
   @optparam icon Caption used for the iconified window.

   If the icon caption is not given, it will default to caption.
*/
FALCON_FUNC sdl_WM_SetCaption ( ::Falcon::VMachine *vm )
{
   Item *i_winName = vm->param(0);
   Item *i_iconName = vm->param(1);

   if ( i_winName == 0 || ! i_winName->isString() ||
        ( i_iconName != 0 && ! i_iconName->isString() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S,[S]" ) ) );
      return;
   }

   if ( i_iconName == 0 )
      i_iconName = i_winName;

   AutoCString caption( *i_winName->asString() );
   AutoCString icon( *i_iconName->asString() );
   ::SDL_WM_SetCaption( caption.c_str(), icon.c_str() );
}

/*#
   @method WM_GetCaption SDL
   @brief Get Caption for SDL window and for window icon.
   @return A two elements array containing the window caption and icon caption.
*/
FALCON_FUNC sdl_WM_GetCaption ( ::Falcon::VMachine *vm )
{
   char *caption, *icon;

   ::SDL_WM_GetCaption( &caption, &icon );
   GarbageString *sCaption = new GarbageString( vm );
   GarbageString *sIcon = new GarbageString( vm );

   if( caption != 0 )
      sCaption->fromUTF8( caption );

   if ( icon != 0 )
      sIcon->fromUTF8( icon );

   CoreArray *array = new CoreArray( vm, 2 );
   array->append( sCaption );
   array->append( sIcon );
   vm->retval( array );
}

/*#
   @method WM_IconifyWindow SDL
   @brief Get Caption for SDL window and for window icon.
   @throw SDLError if the window cannot be iconified.
*/
FALCON_FUNC sdl_WM_IconifyWindow ( ::Falcon::VMachine *vm )
{
   if( SDL_WM_IconifyWindow() == 0 )
   {
      vm->raiseModError( new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 10, __LINE__ )
         .desc( "SDL Iconify Window Error" )
         .extra( SDL_GetError() ) ) );
   }
}

/*#
   @method WM_IconifyWindow SDL
   @optparam grab Grab request.
   @brief Grabs window and mouse input.
   @return Current grab status.

   The request can be one of the following:

   - SDL.GRAB_QUERY - requests current status without changing it.
   - SDL.GRAB_OFF - disable grabbing.
   - SDL.GRAB_ON - enable grabbing.

   Grab request defaults to SDL.GRAB_ON
*/
FALCON_FUNC sdl_WM_GrabInput ( ::Falcon::VMachine *vm )
{
   Item *i_grab = vm->param(0);

   SDL_GrabMode mode;
   if( i_grab == 0 )
   {
      mode = SDL_GRAB_ON;
   }
   else if ( ! i_grab->isInteger() ||
      ( (mode = (SDL_GrabMode) i_grab->asInteger() ) != SDL_GRAB_ON &&
         mode != SDL_GRAB_OFF &&
         mode != SDL_GRAB_QUERY )
   )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "I" ) ) );
      return;
   }

   vm->retval( (int64) SDL_WM_GrabInput( mode ) );
}


}
}

/* end of sdl_wm_ext.cpp */
