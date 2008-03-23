/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_event_ext.cpp

   Binding for SDL event subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 22 Mar 2008 20:29:06 +0100
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
   Binding for SDL event subsystem.
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

void declare_events( Module *self )
{

   //====================================================
   // EventHandler class
   //
   /*#
      @class SDLEventHandler
      @brief Handles SDL events.

      This class holds a set of callbacks that are meant to be overloaded
      by implementors.
   */
   Falcon::Symbol *c_eventhandler = self->addClass( "SDLEventHandler" );
   self->addClassMethod( c_eventhandler, "PollEvent", Falcon::Ext::SDLEventHandler_PollEvent );
   self->addClassMethod( c_eventhandler, "WaitEvent", Falcon::Ext::SDLEventHandler_WaitEvent );
   self->addClassMethod( c_eventhandler, "PushEvent", Falcon::Ext::SDLEventHandler_PushEvent );
   self->addClassMethod( c_eventhandler, "PushUserEvent", Falcon::Ext::SDLEventHandler_PushUserEvent );
   /*#
      @method onActive SDLEventHandler
      @param
      @brief Handles SDL events.
   */

}


void internal_dispatchEvent( VMachine *vm, SDL_Event &evt )
{
   Item method;
   uint32 params;

   CoreObject *self = vm->self().asObject();

   switch( evt.type )
   {
      case SDL_ACTIVEEVENT:
         if ( ! self->getMethod( "onActive", method ) )
            return;

         vm->pushParameter( (int64) evt.active.gain );
         vm->pushParameter( (int64) evt.active.state );
         params = 2;
      break;

      case SDL_KEYDOWN:
         if ( ! self->getMethod( "onKeyDown", method ) )
            return;

         vm->pushParameter( (int64) evt.key.state );
         vm->pushParameter( (int64) evt.key.keysym.scancode );
         vm->pushParameter( (int64) evt.key.keysym.sym );
         vm->pushParameter( (int64) evt.key.keysym.mod );
         vm->pushParameter( (int64) evt.key.keysym.unicode );
         params = 5;
      break;

      case SDL_KEYUP:
         if ( ! self->getMethod( "onKeyUp", method ) )
            return;

         vm->pushParameter( (int64) evt.key.state );
         vm->pushParameter( (int64) evt.key.keysym.scancode );
         vm->pushParameter( (int64) evt.key.keysym.sym );
         vm->pushParameter( (int64) evt.key.keysym.mod );
         vm->pushParameter( (int64) evt.key.keysym.unicode );
         params = 5;
      break;

      case SDL_MOUSEMOTION:
         if ( ! self->getMethod( "onMouseMotion", method ) )
            return;


         vm->pushParameter( (int64) evt.motion.state );
         vm->pushParameter( (int64) evt.motion.x );
         vm->pushParameter( (int64) evt.motion.y );
         vm->pushParameter( (int64) evt.motion.xrel );
         vm->pushParameter( (int64) evt.motion.yrel );

         params = 5;
      break;

      case SDL_MOUSEBUTTONDOWN:
         if ( ! self->getMethod( "onMouseButtonDown", method ) )
            return;

         vm->pushParameter( (int64) evt.button.button );
         vm->pushParameter( (int64) evt.button.state );
         vm->pushParameter( (int64) evt.button.x );
         vm->pushParameter( (int64) evt.button.y );
         params = 4;
      break;

      case SDL_MOUSEBUTTONUP:
         if ( ! self->getMethod( "onMouseButtonUp", method ) )
            return;

         vm->pushParameter( (int64) evt.button.button );
         vm->pushParameter( (int64) evt.button.state );
         vm->pushParameter( (int64) evt.button.x );
         vm->pushParameter( (int64) evt.button.y );
         params = 4;
      break;

      case SDL_JOYAXISMOTION:
         if ( ! self->getMethod( "onJoyAxisMotion", method ) )
            return;

         vm->pushParameter( (int64) evt.jaxis.which );
         vm->pushParameter( (int64) evt.jaxis.axis );
         vm->pushParameter( (int64) evt.jaxis.value );
         params = 3;
      break;

      case SDL_JOYBALLMOTION:
         if ( ! self->getMethod( "onJoyBallMotion", method ) )
            return;

         vm->pushParameter( (int64) evt.jball.which );
         vm->pushParameter( (int64) evt.jball.ball );
         vm->pushParameter( (int64) evt.jball.xrel );
         vm->pushParameter( (int64) evt.jball.yrel );
         params = 4;
      break;

      case SDL_JOYHATMOTION:
         if ( ! self->getMethod( "onJoyHat", method ) )
            return;

         vm->pushParameter( (int64) evt.jhat.which );
         vm->pushParameter( (int64) evt.jhat.hat );
         vm->pushParameter( (int64) evt.jhat.value );
         params = 3;
      break;

      case SDL_JOYBUTTONDOWN:
         if ( ! self->getMethod( "onJoyButtonDown", method ) )
            return;

         vm->pushParameter( (int64) evt.jbutton.which );
         vm->pushParameter( (int64) evt.jbutton.button );
         vm->pushParameter( (int64) evt.jbutton.state );
         params = 3;
      break;

      case SDL_JOYBUTTONUP:
         if ( ! self->getMethod( "onJoyButtonUp", method ) )
            return;

         vm->pushParameter( (int64) evt.jbutton.which );
         vm->pushParameter( (int64) evt.jbutton.button );
         vm->pushParameter( (int64) evt.jbutton.state );
         params = 3;
      break;

      case SDL_VIDEORESIZE:
         if ( ! self->getMethod( "onResize", method ) )
            return;

         vm->pushParameter( (int64) evt.resize.w );
         vm->pushParameter( (int64) evt.resize.h );
         params = 2;
      break;

      case SDL_VIDEOEXPOSE:
         if ( ! self->getMethod( "onExpose", method ) )
            return;

         params = 0;
      break;

      case SDL_QUIT:
         if ( ! self->getMethod( "onQuit", method ) )
            return;

         params = 0;
      break;

      case SDL_NUMEVENTS-1:
         // we must anyhow unlock the garbage data if it's not zero
         if ( ! self->getMethod( "onUserEvent", method ) )
         {
            GarbageLock *lock = (GarbageLock *) evt.user.data1;
            if ( lock != 0 )
               vm->memPool()->unlock( lock );
            return;
         }

         vm->pushParameter( (int64) evt.user.code );
         if( evt.user.data1 == 0 )
         {
            params = 1;
         }
         else {
            GarbageLock *lock = (GarbageLock *) evt.user.data1;
            vm->pushParameter( lock->item() );
            vm->memPool()->unlock( lock );
            params = 2;
         }
      break;
   }

   vm->callFrame( method, params );
}


FALCON_FUNC SDLEventHandler_PollEvent( VMachine *vm )
{
   SDL_Event evt;
   int res = SDL_PollEvent( &evt );
   if ( res == 1 )
   {
      internal_dispatchEvent( vm, evt );
   }

   vm->retval( (int64) res );
}



bool SDLEventHandler_WaitEvent_next( VMachine *vm )
{
   SDL_Event evt;

   int res = SDL_PollEvent( &evt );
   if ( res == 1 )
   {
      internal_dispatchEvent( vm, evt );
      // we're done -- but we have still a call pending

      vm->retval( (int64) 1 );
      vm->returnHandler( 0 );  // do not call us anymore
      return true;
   }
   else {
      // prepare to try again after a yield
      vm->yieldRequest( 0.001 );
      return true;
   }
}

FALCON_FUNC SDLEventHandler_WaitEvent( VMachine *vm )
{
   SDL_Event evt;
   int res = SDL_PollEvent( &evt );
   if ( res == 1 )
   {
      internal_dispatchEvent( vm, evt );
      vm->retval( (int64) 1 );
   }
   else {
      // prepare to try again after a yield
      vm->returnHandler( SDLEventHandler_WaitEvent_next );
      vm->yieldRequest( 0.001 );
   }
}


FALCON_FUNC SDLEventHandler_PushEvent( VMachine *vm )
{
}

/*#
   @method PushUserEvent SDLEventHandler
   @param code A numeric code that should tell what the receiver should do.
   @optparam data any item or object that will be passed to the receiver.
   @return true on success, false if the event queue is full

   When a PollEvent or WaitEvent method is issued on an handler, the
   onUserData
   @note This is a static method. It can be called on the SDLEventHandler class.
*/

FALCON_FUNC SDLEventHandler_PushUserEvent( VMachine *vm )
{
   Item *i_code = vm->param( 0 );
   Item *i_user_data = vm->param( 1 );

   if( i_code == 0 || ! i_code->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,[X]" ) ) );
      return;
   }

   SDL_Event evt;
   evt.type = SDL_NUMEVENTS-1;
   evt.user.code = (int) i_code->forceInteger();
   GarbageLock *lock = 0;

   if( i_user_data != 0 )
   {
      evt.user.data1 = lock = vm->memPool()->lock( *i_user_data );
   }

   if ( ::SDL_PushEvent( &evt ) == 0 )
      vm->retval( true );
   else {
      if ( lock != 0 )
         vm->memPool()->unlock( lock );
      vm->retval( false );
   }
}

}
}
