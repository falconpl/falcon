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

   //====================================================
   // EventType enumeration
   //
   /*#
      @class SDLEventType
      @brief Enumeration of SDL event types.

      This enumeration contains the following types:

      - ACTIVEEVENT
      - KEYDOWN
      - KEYUP
      - MOUSEMOTION
      - MOUSEBUTTONDOWN
      - MOUSEBUTTONUP
      - JOYAXISMOTION
      - JOYBALLMOTION
      - JOYHATMOTION
      - JOYBUTTONDOWN
      - JOYBUTTONUP
      - VIDEORESIZE
      - VIDEOEXPOSE
      - QUIT
   */
   Falcon::Symbol *c_evttype = self->addClass( "SDLEventType" );
   self->addClassProperty( c_evttype, "ACTIVEEVENT" )->setInteger( SDL_ACTIVEEVENT );
   self->addClassProperty( c_evttype, "KEYDOWN" )->setInteger( SDL_KEYDOWN );
   self->addClassProperty( c_evttype, "KEYUP" )->setInteger( SDL_KEYUP );
   self->addClassProperty( c_evttype, "MOUSEMOTION" )->setInteger( SDL_MOUSEMOTION );
   self->addClassProperty( c_evttype, "MOUSEBUTTONDOWN" )->setInteger( SDL_MOUSEBUTTONDOWN );
   self->addClassProperty( c_evttype, "MOUSEBUTTONUP" )->setInteger( SDL_MOUSEBUTTONUP );
   self->addClassProperty( c_evttype, "JOYAXISMOTION" )->setInteger( SDL_JOYAXISMOTION );
   self->addClassProperty( c_evttype, "JOYBALLMOTION" )->setInteger( SDL_JOYBALLMOTION );
   self->addClassProperty( c_evttype, "JOYHATMOTION" )->setInteger( SDL_JOYHATMOTION );
   self->addClassProperty( c_evttype, "JOYBUTTONDOWN" )->setInteger( SDL_JOYBUTTONDOWN );
   self->addClassProperty( c_evttype, "JOYBUTTONUP" )->setInteger( SDL_JOYBUTTONUP );
   self->addClassProperty( c_evttype, "VIDEORESIZE" )->setInteger( SDL_VIDEORESIZE );
   self->addClassProperty( c_evttype, "VIDEOEXPOSE" )->setInteger( SDL_VIDEOEXPOSE );
   self->addClassProperty( c_evttype, "QUIT" )->setInteger( SDL_QUIT );

}

/*#
   @method onActive SDLEventHandler
   @brief Application visibility event handler.
   @param gain 0 if the event is a loss or 1 if it is a gain.
   @param state SDL.APPMOUSEFOCUS if mouse focus was gained
         or lost, SDL.APPINPUTFOCUS if input focus was gained or lost,
         or SDL.APPACTIVE if the application was iconified (gain=0) or
         restored (gain=1).

   See SDL_ActiveEvent description in SDL documentation.

   Overload this method to receive ActiveEvent notifications.
*/

/*#
   @method onKeyDown SDLEventHandler
   @brief Keyboard key down event handler.
   @param state SDL.PRESSED or SDL.RELEASED
   @param scancode  Hardware specific scancode
   @param sym  SDL virtual keysym
   @param mod  Current key modifiers
   @param unicode  Translated character

   See SDL_KeyboardEvent description in SDL documentation.

   Overload this method to receive SDL_KEYDOWN notifications.
*/

/*#
   @method onKeyUp SDLEventHandler
   @brief Keyboard key down event handler.
   @param state SDL.PRESSED or SDL.RELEASED
   @param scancode  Hardware specific scancode
   @param sym  SDL virtual keysym
   @param mod  Current key modifiers
   @param unicode  Translated character

   See SDL_KeyboardEvent description in SDL documentation.

   Overload this method to receive SDL_KEYUP notifications.
*/

/*#
   @method onMouseMotion SDLEventHandler
   @brief  Mouse motion event handler
   @param state The current button state
   @param x  X coordinate of the mouse
   @param y  X coordinate of the mouse
   @param xrel relative movement of mouse on the X axis with respect to last notification.
   @param yrel relative movement of mouse on the X axis with respect to last notification.

   See SDL_MouseMotionEvent description in SDL documentation.

   Overload this method to receive SDL_MOUSEMOTION notifications.
*/

/*#
   @method onMouseButtonDown SDLEventHandler
   @brief  Mouse button event handler
   @param state The current button state
   @param button The mouse button index (SDL_BUTTON_LEFT, SDL_BUTTON_MIDDLE, SDL_BUTTON_RIGHT)
   @param x X coordinate of the mouse
   @param y X coordinate of the mouse

   See SDL_MouseButtonEvent description in SDL documentation.

   Overload this method to receive SDL_MOUSEBUTTONDOWN notifications.
*/

/*#
   @method onMouseButtonUp SDLEventHandler
   @brief  Mouse button event handler
   @param state The current button state
   @param button The mouse button index (SDL_BUTTON_LEFT, SDL_BUTTON_MIDDLE, SDL_BUTTON_RIGHT)
   @param x X coordinate of the mouse
   @param y X coordinate of the mouse

   See SDL_MouseButtonEvent description in SDL documentation.

   Overload this method to receive SDL_MOUSEBUTTONUP notifications.
*/

/*#
   @method onJoyAxisMotion SDLEventHandler
   @brief  Joystick axis motion event handler
   @param which  Joystick device index
   @param axis  Joystick axis index
   @param value  Axis value (range: -32768 to 32767)

   See SDL_JoyAxisEvent description in SDL documentation.

   Overload this method to receive SDL_JOYAXISMOTION notifications.
*/

/*#
   @method onJoyButtonDown SDLEventHandler
   @brief  Joystick button event handler
   @param which  Joystick device index
   @param button  Joystick button index
   @param state  SDL_PRESSED or SDL_RELEASED

   See SDL_JoyButtonEvent description in SDL documentation.

   Overload this method to receive SDL_JOYBUTTONDOWN notifications.
*/

/*#
   @method onJoyButtonUp SDLEventHandler
   @brief  Joystick button event handler
   @param which  Joystick device index
   @param button  Joystick button index
   @param state  SDL_PRESSED or SDL_RELEASED

   See SDL_JoyButtonEvent description in SDL documentation.

   Overload this method to receive SDL_JOYBUTTONUP notifications.
*/

/*#
   @method onJoyHatMotion SDLEventHandler
   @brief Joystick hat position change event handler
   @param which  Joystick device index
   @param hat  Joystick hat index
   @param value  hat position.

   See SDL_JoyHatEvent description in SDL documentation.

   Overload this method to receive SDL_JOYHATMOTION notifications.
*/

/*#
   @method onJoyBallMotion SDLEventHandler
   @brief Joystick trackball motion event handler
   @param which  Joystick device index
   @param ball  Joystick trackball index
   @param xrel  The relative motion in the X direction
   @param yrel  The relative motion in the Y direction

   See SDL_JoyBallEvent description in SDL documentation.

   Overload this method to receive SDL_JOYBALLMOTION notifications.
*/

/*#
   @method onResize SDLEventHandler
   @brief Window resize event handler
   @param w  New width of the window
   @param h  New height of the window

   See SDL_ResizeEvent description in SDL documentation.

   Overload this method to receive SDL_VIDEORESIZE notifications.
*/

/*#
   @method onExpose SDLEventHandler
   @brief Window exposition (need redraw) notification.

   See SDL_ExposeEvent description in SDL documentation.

   Overload this method to receive SDL_VIDEOEXPOSE notifications.
*/

/*#
   @method onQuit SDLEventHandler
   @brief Quit requested event.

   See SDL_Quit description in SDL documentation.
   This notification means that the user asked the application to terminate.
   The application should call exit(0) if no other cleanup routines are needed,
   or perform cleanup and notifications for clean exit to other threads/coroutines.
*/

/*#
   @method onUserEvent SDLEventHandler
   @brief Receives custom events generated by the application.
   @param code An arbitrary integer code available for the application.
   @param item An arbitrary item that can be passed to the handler.

   Applications can generate user events through SDL.PostUserEvent or
   the static method @a SDLEventHandler.PostUserEvent (they are the same).

   Optionally, those calls can pass an arbitrary item of any type or class
   to the handler; the handler can inspect it and even manipulate it.
*/

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
         if ( ! self->getMethod( "onJoyHatMotion", method ) )
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

/*#
   @method PollEvent SDLEventHandler
   @brief Polls for event and calls handlers if events are incoming.
   @return true if an event has been processed.

   This method checks the SDL event queue, and if an event is ready
   for processing, it calls the handler provided by this instance.

   To provide event handlers, it is necessary to derive a subclass
   from SDLEventHandler overloading the callbacks that must be
   handled, and then call PollEvent or WaitEvent on the instance.

   If there isn't any event to be processed, this method returns
   immediately false.
*/

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

/*#
   @method WaitEvent SDLEventHandler
   @brief Waits forever until an event is received.

   This method blocks the current coroutine until a SDL event is
   received. However, the VM is able to proceed with other
   coroutines.

   To unblock this wait from another coroutine, just post an unprocessable
   event through @a SDLEventHandler.PushUserEvent.

   As soon as a message is received and processed, the function returns, so
   it is possible to set a global flag in the message processors to communicate
   new program status to the subsequent code.

   In example, the following is a minimal responsive SDL Falcon application.

   \code
      object handler from SDLEventHandler
         shouldQuit = false

         function onQuit()
            self.shouldQuit = true
         end
      end

      ...
      ...
      // main code
      while not handler.shouldQuit
         handler.WaitEvent()
      end
   \endcode
*/
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

/*#
   @method PushEvent SDLEventHandler
   @param type one of the SDL events.
   @optparam ... Other parameters that vary depending on the event type
   @return true on success, false if the event queue is full

   This is a static method.
   @todo
*/

FALCON_FUNC SDLEventHandler_PushEvent( VMachine *vm )
{
}

/*#
   @method PushUserEvent SDLEventHandler
   @param code A numeric code that should tell what the receiver should do.
   @optparam data any item or object that will be passed to the receiver.
   @return true on success, false if the event queue is full

   After this call is pefrormed when a PollEvent or WaitEvent method is issued
   on an event handler of class @a SDLEventHandler, the onUserData method of that handler is
   invoked (if provided).

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

//=================================================================
// Generic event mangling
//

/*#
   @method EventState SDL
   @brief Query or change processing ability.
   @param type Event type to be filtered.
   @param state Wether to enable, disable or query the event state.
   @return true on success, false if the event queue is full

   When a PollEvent or WaitEvent method is issued on an handler, the
   onUserData
   @note This is a static method. It can be called on the SDLEventHandler class.
*/
FALCON_FUNC SDL_EventState( VMachine *vm )
{

}

}
}
