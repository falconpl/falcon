/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_event_ext.cpp

   Binding for SDL event subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 22 Mar 2008 20:29:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Binding for SDL event subsystem.
*/

#define FALCON_EXPORT_SERVICE

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>
#include <falcon/garbagelock.h>
#include <falcon/mt.h>
#include <falcon/vmmsg.h>

#include "sdl_ext.h"
#include "sdl_mod.h"

#include <SDL.h>

/*#
   @beginmodule sdl
*/


namespace Falcon {
namespace Ext {

#if ! (defined(WIN32) || defined(FALCON_SYSTEM_MAC)) 
   #define USE_SDL_EVENT_THREADS
#endif

FALCON_FUNC _coroutinePoll( VMachine *vm );

void declare_events( Module *self )
{


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
   self->addClassProperty( c_evttype, "ACTIVEEVENT" ).setInteger( SDL_ACTIVEEVENT );
   self->addClassProperty( c_evttype, "KEYDOWN" ).setInteger( SDL_KEYDOWN );
   self->addClassProperty( c_evttype, "KEYUP" ).setInteger( SDL_KEYUP );
   self->addClassProperty( c_evttype, "MOUSEMOTION" ).setInteger( SDL_MOUSEMOTION );
   self->addClassProperty( c_evttype, "MOUSEBUTTONDOWN" ).setInteger( SDL_MOUSEBUTTONDOWN );
   self->addClassProperty( c_evttype, "MOUSEBUTTONUP" ).setInteger( SDL_MOUSEBUTTONUP );
   self->addClassProperty( c_evttype, "JOYAXISMOTION" ).setInteger( SDL_JOYAXISMOTION );
   self->addClassProperty( c_evttype, "JOYBALLMOTION" ).setInteger( SDL_JOYBALLMOTION );
   self->addClassProperty( c_evttype, "JOYHATMOTION" ).setInteger( SDL_JOYHATMOTION );
   self->addClassProperty( c_evttype, "JOYBUTTONDOWN" ).setInteger( SDL_JOYBUTTONDOWN );
   self->addClassProperty( c_evttype, "JOYBUTTONUP" ).setInteger( SDL_JOYBUTTONUP );
   self->addClassProperty( c_evttype, "VIDEORESIZE" ).setInteger( SDL_VIDEORESIZE );
   self->addClassProperty( c_evttype, "VIDEOEXPOSE" ).setInteger( SDL_VIDEOEXPOSE );
   self->addClassProperty( c_evttype, "QUIT" ).setInteger( SDL_QUIT );

   #ifndef USE_SDL_EVENT_THREADS
   const Symbol* coropoll = self->addExtFunc( "_coroutinePoll", _coroutinePoll, false );
   coropoll->setWKS( true );
   #endif

   //====================================================
   // EventType enumeration
   //
   /*#
      @class SDLK
      @brief Enumeration for SDL Scan Key codes.

      This enumeration contains the scan key codes that are
      returned by SDL when querying for a certain key.

      These are the values that are stored in this enumeration:

      - SDLK.BACKSPACE: backspace
      - SDLK.TAB: tab
      - SDLK.CLEAR: clear
      - SDLK.RETURN: return
      - SDLK.PAUSE: pause
      - SDLK.ESCAPE: escape
      - SDLK.SPACE: space
      - SDLK.EXCLAIM: exclaim
      - SDLK.QUOTEDBL: quotedbl
      - SDLK.HASH: hash
      - SDLK.DOLLAR: dollar
      - SDLK.AMPERSAND: ampersand
      - SDLK.QUOTE: quote
      - SDLK.LEFTPAREN: left parenthesis
      - SDLK.RIGHTPAREN: right parenthesis
      - SDLK.ASTERISK: asterisk
      - SDLK.PLUS: plus sign
      - SDLK.COMMA: comma
      - SDLK.MINUS: minus sign
      - SDLK.PERIOD: period
      - SDLK.SLASH: forward slash
      - SDLK.0: 0
      - SDLK.1: 1
      - SDLK.2: 2
      - SDLK.3: 3
      - SDLK.4: 4
      - SDLK.5: 5
      - SDLK.6: 6
      - SDLK.7: 7
      - SDLK.8: 8
      - SDLK.9: 9
      - SDLK.COLON: colon
      - SDLK.SEMICOLON: semicolon
      - SDLK.LESS: less-than sign
      - SDLK.EQUALS: equals sign
      - SDLK.GREATER: greater-than sign
      - SDLK.QUESTION: question mark
      - SDLK.AT: at
      - SDLK.LEFTBRACKET: left bracket
      - SDLK.BACKSLASH: backslash
      - SDLK.RIGHTBRACKET: right bracket
      - SDLK.CARET: caret
      - SDLK.UNDERSCORE: underscore
      - SDLK.BACKQUOTE: grave
      - SDLK.a: a
      - SDLK.b: b
      - SDLK.c: c
      - SDLK.d: d
      - SDLK.e: e
      - SDLK.f: f
      - SDLK.g: g
      - SDLK.h: h
      - SDLK.i: i
      - SDLK.j: j
      - SDLK.k: k
      - SDLK.l: l
      - SDLK.m: m
      - SDLK.n: n
      - SDLK.o: o
      - SDLK.p: p
      - SDLK.q: q
      - SDLK.r: r
      - SDLK.s: s
      - SDLK.t: t
      - SDLK.u: u
      - SDLK.v: v
      - SDLK.w: w
      - SDLK.x: x
      - SDLK.y: y
      - SDLK.z: z
      - SDLK.DELETE: delete
      - SDLK.KP0: keypad 0
      - SDLK.KP1: keypad 1
      - SDLK.KP2: keypad 2
      - SDLK.KP3: keypad 3
      - SDLK.KP4: keypad 4
      - SDLK.KP5: keypad 5
      - SDLK.KP6: keypad 6
      - SDLK.KP7: keypad 7
      - SDLK.KP8: keypad 8
      - SDLK.KP9: keypad 9
      - SDLK.KP_PERIOD: keypad period
      - SDLK.KP_DIVIDE: keypad divide
      - SDLK.KP_MULTIPLY: keypad multiply
      - SDLK.KP_MINUS: keypad minus
      - SDLK.KP_PLUS: keypad plus
      - SDLK.KP_ENTER: keypad enter
      - SDLK.KP_EQUALS: keypad equals
      - SDLK.UP: up arrow
      - SDLK.DOWN: down arrow
      - SDLK.RIGHT: right arrow
      - SDLK.LEFT: left arrow
      - SDLK.INSERT: insert
      - SDLK.HOME: home
      - SDLK.END: end
      - SDLK.PAGEUP: page up
      - SDLK.PAGEDOWN: page down
      - SDLK.F1: F1
      - SDLK.F2: F2
      - SDLK.F3: F3
      - SDLK.F4: F4
      - SDLK.F5: F5
      - SDLK.F6: F6
      - SDLK.F7: F7
      - SDLK.F8: F8
      - SDLK.F9: F9
      - SDLK.F10: F10
      - SDLK.F11: F11
      - SDLK.F12: F12
      - SDLK.F13: F13
      - SDLK.F14: F14
      - SDLK.F15: F15
      - SDLK.NUMLOCK: numlock
      - SDLK.CAPSLOCK: capslock
      - SDLK.SCROLLOCK: scrollock
      - SDLK.RSHIFT: right shift
      - SDLK.LSHIFT: left shift
      - SDLK.RCTRL: right ctrlTEST_PARA_CORO
      - SDLK.LCTRL: left ctrl
      - SDLK.RALT: right alt
      - SDLK.LALT: left alt
      - SDLK.RMETA: right meta
      - SDLK.LMETA: left meta
      - SDLK.LSUPER: left windows key
      - SDLK.RSUPER: right windows key
      - SDLK.MODE: mode shift
      - SDLK.HELP: help
      - SDLK.PRINT: print-screen
      - SDLK.SYSREQ: SysRq
      - SDLK.BREAK: break
      - SDLK.MENU: menu
      - SDLK.POWER: power
      - SDLK.EURO: euro
   */
   Falcon::Symbol *c_sdlk = self->addClass( "SDLK" );
   self->addClassProperty( c_sdlk, "BACKSPACE" ).setInteger( SDLK_BACKSPACE );
   self->addClassProperty( c_sdlk, "TAB" ).setInteger( SDLK_TAB );
   self->addClassProperty( c_sdlk, "CLEAR" ).setInteger( SDLK_CLEAR );
   self->addClassProperty( c_sdlk, "RETURN" ).setInteger( SDLK_RETURN );
   self->addClassProperty( c_sdlk, "PAUSE" ).setInteger( SDLK_PAUSE );
   self->addClassProperty( c_sdlk, "ESCAPE" ).setInteger( SDLK_ESCAPE );
   self->addClassProperty( c_sdlk, "SPACE" ).setInteger( SDLK_SPACE );
   self->addClassProperty( c_sdlk, "EXCLAIM" ).setInteger( SDLK_EXCLAIM );
   self->addClassProperty( c_sdlk, "QUOTEDBL" ).setInteger( SDLK_QUOTEDBL );
   self->addClassProperty( c_sdlk, "HASH" ).setInteger( SDLK_HASH );
   self->addClassProperty( c_sdlk, "DOLLAR" ).setInteger( SDLK_DOLLAR );
   self->addClassProperty( c_sdlk, "AMPERSAND" ).setInteger( SDLK_AMPERSAND );
   self->addClassProperty( c_sdlk, "QUOTE" ).setInteger( SDLK_QUOTE );
   self->addClassProperty( c_sdlk, "LEFTPAREN" ).setInteger( SDLK_LEFTPAREN );
   self->addClassProperty( c_sdlk, "RIGHTPAREN" ).setInteger( SDLK_RIGHTPAREN );
   self->addClassProperty( c_sdlk, "ASTERISK" ).setInteger( SDLK_ASTERISK );
   self->addClassProperty( c_sdlk, "PLUS" ).setInteger( SDLK_PLUS );
   self->addClassProperty( c_sdlk, "COMMA" ).setInteger( SDLK_COMMA );
   self->addClassProperty( c_sdlk, "MINUS" ).setInteger( SDLK_MINUS );
   self->addClassProperty( c_sdlk, "PERIOD" ).setInteger( SDLK_PERIOD );
   self->addClassProperty( c_sdlk, "SLASH" ).setInteger( SDLK_SLASH );
   self->addClassProperty( c_sdlk, "0" ).setInteger( SDLK_0 );
   self->addClassProperty( c_sdlk, "1" ).setInteger( SDLK_1 );
   self->addClassProperty( c_sdlk, "2" ).setInteger( SDLK_2 );
   self->addClassProperty( c_sdlk, "3" ).setInteger( SDLK_3 );
   self->addClassProperty( c_sdlk, "4" ).setInteger( SDLK_4 );
   self->addClassProperty( c_sdlk, "5" ).setInteger( SDLK_5 );
   self->addClassProperty( c_sdlk, "6" ).setInteger( SDLK_6 );
   self->addClassProperty( c_sdlk, "7" ).setInteger( SDLK_7 );
   self->addClassProperty( c_sdlk, "8" ).setInteger( SDLK_8 );
   self->addClassProperty( c_sdlk, "9" ).setInteger( SDLK_9 );
   self->addClassProperty( c_sdlk, "COLON" ).setInteger( SDLK_COLON );
   self->addClassProperty( c_sdlk, "SEMICOLON" ).setInteger( SDLK_SEMICOLON );
   self->addClassProperty( c_sdlk, "LESS" ).setInteger( SDLK_LESS );
   self->addClassProperty( c_sdlk, "EQUALS" ).setInteger( SDLK_EQUALS );
   self->addClassProperty( c_sdlk, "GREATER" ).setInteger( SDLK_GREATER );
   self->addClassProperty( c_sdlk, "QUESTION" ).setInteger( SDLK_QUESTION );
   self->addClassProperty( c_sdlk, "AT" ).setInteger( SDLK_AT );
   self->addClassProperty( c_sdlk, "LEFTBRACKET" ).setInteger( SDLK_LEFTBRACKET );
   self->addClassProperty( c_sdlk, "BACKSLASH" ).setInteger( SDLK_BACKSLASH );
   self->addClassProperty( c_sdlk, "RIGHTBRACKET" ).setInteger( SDLK_RIGHTBRACKET );
   self->addClassProperty( c_sdlk, "CARET" ).setInteger( SDLK_CARET );
   self->addClassProperty( c_sdlk, "UNDERSCORE" ).setInteger( SDLK_UNDERSCORE );
   self->addClassProperty( c_sdlk, "BACKQUOTE" ).setInteger( SDLK_BACKQUOTE );
   self->addClassProperty( c_sdlk, "a" ).setInteger( SDLK_a );
   self->addClassProperty( c_sdlk, "b" ).setInteger( SDLK_b );
   self->addClassProperty( c_sdlk, "c" ).setInteger( SDLK_c );
   self->addClassProperty( c_sdlk, "d" ).setInteger( SDLK_d );
   self->addClassProperty( c_sdlk, "e" ).setInteger( SDLK_e );
   self->addClassProperty( c_sdlk, "f" ).setInteger( SDLK_f );
   self->addClassProperty( c_sdlk, "g" ).setInteger( SDLK_g );
   self->addClassProperty( c_sdlk, "h" ).setInteger( SDLK_h );
   self->addClassProperty( c_sdlk, "i" ).setInteger( SDLK_i );
   self->addClassProperty( c_sdlk, "j" ).setInteger( SDLK_j );
   self->addClassProperty( c_sdlk, "k" ).setInteger( SDLK_k );
   self->addClassProperty( c_sdlk, "l" ).setInteger( SDLK_l );
   self->addClassProperty( c_sdlk, "m" ).setInteger( SDLK_m );
   self->addClassProperty( c_sdlk, "n" ).setInteger( SDLK_n );
   self->addClassProperty( c_sdlk, "o" ).setInteger( SDLK_o );
   self->addClassProperty( c_sdlk, "p" ).setInteger( SDLK_p );
   self->addClassProperty( c_sdlk, "q" ).setInteger( SDLK_q );
   self->addClassProperty( c_sdlk, "r" ).setInteger( SDLK_r );
   self->addClassProperty( c_sdlk, "s" ).setInteger( SDLK_s );
   self->addClassProperty( c_sdlk, "t" ).setInteger( SDLK_t );
   self->addClassProperty( c_sdlk, "u" ).setInteger( SDLK_u );
   self->addClassProperty( c_sdlk, "v" ).setInteger( SDLK_v );
   self->addClassProperty( c_sdlk, "w" ).setInteger( SDLK_w );
   self->addClassProperty( c_sdlk, "x" ).setInteger( SDLK_x );
   self->addClassProperty( c_sdlk, "y" ).setInteger( SDLK_y );
   self->addClassProperty( c_sdlk, "z" ).setInteger( SDLK_z );
   self->addClassProperty( c_sdlk, "DELETE" ).setInteger( SDLK_DELETE );
   self->addClassProperty( c_sdlk, "KP0" ).setInteger( SDLK_KP0 );
   self->addClassProperty( c_sdlk, "KP1" ).setInteger( SDLK_KP1 );
   self->addClassProperty( c_sdlk, "KP2" ).setInteger( SDLK_KP2 );
   self->addClassProperty( c_sdlk, "KP3" ).setInteger( SDLK_KP3 );
   self->addClassProperty( c_sdlk, "KP4" ).setInteger( SDLK_KP4 );
   self->addClassProperty( c_sdlk, "KP5" ).setInteger( SDLK_KP5 );
   self->addClassProperty( c_sdlk, "KP6" ).setInteger( SDLK_KP6 );
   self->addClassProperty( c_sdlk, "KP7" ).setInteger( SDLK_KP7 );
   self->addClassProperty( c_sdlk, "KP8" ).setInteger( SDLK_KP8 );
   self->addClassProperty( c_sdlk, "KP9" ).setInteger( SDLK_KP9 );
   self->addClassProperty( c_sdlk, "KP_PERIOD" ).setInteger( SDLK_KP_PERIOD );
   self->addClassProperty( c_sdlk, "KP_DIVIDE" ).setInteger( SDLK_KP_DIVIDE );
   self->addClassProperty( c_sdlk, "KP_MULTIPLY" ).setInteger( SDLK_KP_MULTIPLY );
   self->addClassProperty( c_sdlk, "KP_MINUS" ).setInteger( SDLK_KP_MINUS );
   self->addClassProperty( c_sdlk, "KP_PLUS" ).setInteger( SDLK_KP_PLUS );
   self->addClassProperty( c_sdlk, "KP_ENTER" ).setInteger( SDLK_KP_ENTER );
   self->addClassProperty( c_sdlk, "KP_EQUALS" ).setInteger( SDLK_KP_EQUALS );
   self->addClassProperty( c_sdlk, "UP" ).setInteger( SDLK_UP );
   self->addClassProperty( c_sdlk, "DOWN" ).setInteger( SDLK_DOWN );
   self->addClassProperty( c_sdlk, "RIGHT" ).setInteger( SDLK_RIGHT );
   self->addClassProperty( c_sdlk, "LEFT" ).setInteger( SDLK_LEFT );
   self->addClassProperty( c_sdlk, "INSERT" ).setInteger( SDLK_INSERT );
   self->addClassProperty( c_sdlk, "HOME" ).setInteger( SDLK_HOME );
   self->addClassProperty( c_sdlk, "END" ).setInteger( SDLK_END );
   self->addClassProperty( c_sdlk, "PAGEUP" ).setInteger( SDLK_PAGEUP );
   self->addClassProperty( c_sdlk, "PAGEDOWN" ).setInteger( SDLK_PAGEDOWN );
   self->addClassProperty( c_sdlk, "F1" ).setInteger( SDLK_F1 );
   self->addClassProperty( c_sdlk, "F2" ).setInteger( SDLK_F2 );
   self->addClassProperty( c_sdlk, "F3" ).setInteger( SDLK_F3 );
   self->addClassProperty( c_sdlk, "F4" ).setInteger( SDLK_F4 );
   self->addClassProperty( c_sdlk, "F5" ).setInteger( SDLK_F5 );
   self->addClassProperty( c_sdlk, "F6" ).setInteger( SDLK_F6 );
   self->addClassProperty( c_sdlk, "F7" ).setInteger( SDLK_F7 );
   self->addClassProperty( c_sdlk, "F8" ).setInteger( SDLK_F8 );
   self->addClassProperty( c_sdlk, "F9" ).setInteger( SDLK_F9 );
   self->addClassProperty( c_sdlk, "F10" ).setInteger( SDLK_F10 );
   self->addClassProperty( c_sdlk, "F11" ).setInteger( SDLK_F11 );
   self->addClassProperty( c_sdlk, "F12" ).setInteger( SDLK_F12 );
   self->addClassProperty( c_sdlk, "F13" ).setInteger( SDLK_F13 );
   self->addClassProperty( c_sdlk, "F14" ).setInteger( SDLK_F14 );
   self->addClassProperty( c_sdlk, "F15" ).setInteger( SDLK_F15 );
   self->addClassProperty( c_sdlk, "NUMLOCK" ).setInteger( SDLK_NUMLOCK );
   self->addClassProperty( c_sdlk, "CAPSLOCK" ).setInteger( SDLK_CAPSLOCK );
   self->addClassProperty( c_sdlk, "SCROLLOCK" ).setInteger( SDLK_SCROLLOCK );
   self->addClassProperty( c_sdlk, "RSHIFT" ).setInteger( SDLK_RSHIFT );
   self->addClassProperty( c_sdlk, "LSHIFT" ).setInteger( SDLK_LSHIFT );
   self->addClassProperty( c_sdlk, "RCTRL" ).setInteger( SDLK_RCTRL );
   self->addClassProperty( c_sdlk, "LCTRL" ).setInteger( SDLK_LCTRL );
   self->addClassProperty( c_sdlk, "RALT" ).setInteger( SDLK_RALT );
   self->addClassProperty( c_sdlk, "LALT" ).setInteger( SDLK_LALT );
   self->addClassProperty( c_sdlk, "RMETA" ).setInteger( SDLK_RMETA );
   self->addClassProperty( c_sdlk, "LMETA" ).setInteger( SDLK_LMETA );
   self->addClassProperty( c_sdlk, "LSUPER" ).setInteger( SDLK_LSUPER );
   self->addClassProperty( c_sdlk, "RSUPER" ).setInteger( SDLK_RSUPER );
   self->addClassProperty( c_sdlk, "MODE" ).setInteger( SDLK_MODE );
   self->addClassProperty( c_sdlk, "HELP" ).setInteger( SDLK_HELP );
   self->addClassProperty( c_sdlk, "PRINT" ).setInteger( SDLK_PRINT );
   self->addClassProperty( c_sdlk, "SYSREQ" ).setInteger( SDLK_SYSREQ );
   self->addClassProperty( c_sdlk, "BREAK" ).setInteger( SDLK_BREAK );
   self->addClassProperty( c_sdlk, "MENU" ).setInteger( SDLK_MENU );
   self->addClassProperty( c_sdlk, "POWER" ).setInteger( SDLK_POWER );
   self->addClassProperty( c_sdlk, "EURO" ).setInteger( SDLK_EURO );

   /*#
      @class SDLKMOD
      @brief Enumeration for SDL Key Modifiers.

      This enumeration contains the scan key codes that are
      returned by SDL when querying for a certain key.

      The key modifiers are actually bitfields that can be
      combined through the binary "|" (or) operator.

      Key modifiers in this enumeration are:

      - SDLKMOD.NONE:  No modifiers applicable
      - SDLKMOD.NUM:  Numlock is down
      - SDLKMOD.CAPS:  Capslock is down
      - SDLKMOD.LCTRL:  Left Control is down
      - SDLKMOD.RCTRL:  Right Control is down
      - SDLKMOD.RSHIFT:  Right Shift is down
      - SDLKMOD.LSHIFT:  Left Shift is down
      - SDLKMOD.RALT:  Right Alt is down
      - SDLKMOD.LALT:  Left Alt is down
      - SDLKMOD.RMETA:  Right Meta is down
      - SDLKMOD.LMETA:  Left Meta is down
      - SDLKMOD.CTRL:  A Control key is down ( LCTRL | RCTRL )
      - SDLKMOD.SHIFT:  A Shift key is down ( LSHIFT | RSHIFT )
      - SDLKMOD.ALT:  An Alt key is down ( LALT | RALT )
      - SDLKMOD.META:  A meta key is down ( LMETA | RMETA )
   */
   Falcon::Symbol *c_sdlkmod = self->addClass( "SDLKMOD" );
   self->addClassProperty( c_sdlkmod, "NONE" ).setInteger( KMOD_NONE );
   self->addClassProperty( c_sdlkmod, "NUM" ).setInteger( KMOD_NUM );
   self->addClassProperty( c_sdlkmod, "CAPS" ).setInteger( KMOD_CAPS );
   self->addClassProperty( c_sdlkmod, "LCTRL" ).setInteger( KMOD_LCTRL );
   self->addClassProperty( c_sdlkmod, "RCTRL" ).setInteger( KMOD_RCTRL );
   self->addClassProperty( c_sdlkmod, "RSHIFT" ).setInteger( KMOD_RSHIFT );
   self->addClassProperty( c_sdlkmod, "LSHIFT" ).setInteger( KMOD_LSHIFT );
   self->addClassProperty( c_sdlkmod, "RALT" ).setInteger( KMOD_RALT );
   self->addClassProperty( c_sdlkmod, "LALT" ).setInteger( KMOD_LALT );
   self->addClassProperty( c_sdlkmod, "RMETA" ).setInteger( KMOD_RMETA );
   self->addClassProperty( c_sdlkmod, "LMETA" ).setInteger( KMOD_LMETA );
   self->addClassProperty( c_sdlkmod, "CTRL" ).setInteger( KMOD_CTRL );
   self->addClassProperty( c_sdlkmod, "SHIFT" ).setInteger( KMOD_SHIFT );
   self->addClassProperty( c_sdlkmod, "ALT" ).setInteger( KMOD_ALT );
   self->addClassProperty( c_sdlkmod, "META" ).setInteger( KMOD_META );

   /*#
      @class SDLMouseState
      @brief Allows querying of current mouse status.

      This class is used to wrap mouse event collection functions
      SDL_GetMouseState and SDL_GetRelativeMouseState.

      To query the state of the mouse, it is necessary to create
      an instance of this class, and then call its refresh() method.


      If the program doesn't call periodically event dispatching
      routines or if it doesn't pump events through @a SDL.PumpEvents,
      then it is necessary to use the pumpAndRefresh() method for
      the contents of this class to be updated.

      @prop x Current mouse x position
      @prop y Current mouse y position
      @prop xrel Relative x movement with respect to last check.
      @prop yrel Relative y movement with respect to last check.
      @prop state Mouse button pression state.
   */
   Falcon::Symbol *c_sdlmouse = self->addClass( "SDLMouseState" );
   c_sdlmouse->getClassDef()->factory( &SdlMouseState_Factory );

   sdl_mouse_state mstate;
   self->addClassProperty( c_sdlmouse, "x" ).
      setReflective( Falcon::e_reflectInt, &mstate, &mstate.x ).setReadOnly(true);
   self->addClassProperty( c_sdlmouse, "y" ).
      setReflective( Falcon::e_reflectInt, &mstate, &mstate.y ).setReadOnly(true);
   self->addClassProperty( c_sdlmouse, "xrel" ).
      setReflective( Falcon::e_reflectInt, &mstate, &mstate.xrel ).setReadOnly(true);
   self->addClassProperty( c_sdlmouse, "yrel" ).
      setReflective( Falcon::e_reflectInt, &mstate, &mstate.yrel ).setReadOnly(true);
   self->addClassProperty( c_sdlmouse, "state" ).
      setReflective( Falcon::e_reflectInt, &mstate, &mstate.state ).setReadOnly(true);

   self->addClassMethod( c_sdlmouse, "Refresh", &SDLMouseState_Refresh );
   self->addClassMethod( c_sdlmouse, "PumpAndRefresh", &SDLMouseState_PumpAndRefresh );
}


void internal_dispatchEvent( VMachine *vm, SDL_Event &evt )
{
   Item method;
   uint32 params;
   VMMessage *msg;

   switch( evt.type )
   {
      case SDL_ACTIVEEVENT:
         if ( vm->getSlot( "sdl_Active", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_Active" );
         msg->addParam( (int64) evt.active.gain );
         msg->addParam( (int64) evt.active.state );
      break;

      case SDL_KEYDOWN:
         if ( vm->getSlot( "sdl_KeyDown", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_KeyDown" );
         msg->addParam( (int64) evt.key.state );
         msg->addParam( (int64) evt.key.keysym.scancode );
         msg->addParam( (int64) evt.key.keysym.sym );
         msg->addParam( (int64) evt.key.keysym.mod );
         msg->addParam( (int64) evt.key.keysym.unicode );
      break;

      case SDL_KEYUP:
         if ( vm->getSlot( "sdl_KeyUp", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_KeyUp" );
         msg->addParam( (int64) evt.key.state );
         msg->addParam( (int64) evt.key.keysym.scancode );
         msg->addParam( (int64) evt.key.keysym.sym );
         msg->addParam( (int64) evt.key.keysym.mod );
         msg->addParam( (int64) evt.key.keysym.unicode );
      break;

      case SDL_MOUSEMOTION:
         if ( vm->getSlot( "sdl_MouseMotion", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_MouseMotion" );
         msg->addParam( (int64) evt.motion.state );
         msg->addParam( (int64) evt.motion.x );
         msg->addParam( (int64) evt.motion.y );
         msg->addParam( (int64) evt.motion.xrel );
         msg->addParam( (int64) evt.motion.yrel );
      break;

      case SDL_MOUSEBUTTONDOWN:
         if ( vm->getSlot( "sdl_MouseButtonDown", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_MouseButtonDown" );
         msg->addParam( (int64) evt.button.button );
         msg->addParam( (int64) evt.button.state );
         msg->addParam( (int64) evt.button.x );
         msg->addParam( (int64) evt.button.y );
      break;

      case SDL_MOUSEBUTTONUP:
          if ( vm->getSlot( "sdl_MouseButtonUp", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_MouseButtonUp" );
         msg->addParam( (int64) evt.button.button );
         msg->addParam( (int64) evt.button.state );
         msg->addParam( (int64) evt.button.x );
         msg->addParam( (int64) evt.button.y );
      break;

      case SDL_JOYAXISMOTION:
         if ( vm->getSlot( "sdl_JoyAxisMotion", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_JoyAxisMotion" );
         msg->addParam( (int64) evt.jaxis.which );
         msg->addParam( (int64) evt.jaxis.axis );
         msg->addParam( (int64) evt.jaxis.value );
      break;

      case SDL_JOYBALLMOTION:
         if ( vm->getSlot( "sdl_JoyBallMotion", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_JoyBallMotion" );

         msg->addParam( (int64) evt.jball.which );
         msg->addParam( (int64) evt.jball.ball );
         msg->addParam( (int64) evt.jball.xrel );
         msg->addParam( (int64) evt.jball.yrel );
      break;

      case SDL_JOYHATMOTION:
         if ( vm->getSlot( "sdl_JoyHatMotion", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_JoyHatMotion" );
         msg->addParam( (int64) evt.jhat.which );
         msg->addParam( (int64) evt.jhat.hat );
         msg->addParam( (int64) evt.jhat.value );
      break;

      case SDL_JOYBUTTONDOWN:
         if ( vm->getSlot( "sdl_JoyButtonDown", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_JoyButtonDown" );
         msg->addParam( (int64) evt.jbutton.which );
         msg->addParam( (int64) evt.jbutton.button );
         msg->addParam( (int64) evt.jbutton.state );
      break;

      case SDL_JOYBUTTONUP:
         if ( vm->getSlot( "sdl_JoyButtonUp", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_JoyButtonUp" );

         msg->addParam( (int64) evt.jbutton.which );
         msg->addParam( (int64) evt.jbutton.button );
         msg->addParam( (int64) evt.jbutton.state );
      break;

      case SDL_VIDEORESIZE:
         if ( vm->getSlot( "sdl_Resize", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_Resize" );

         msg->addParam( (int64) evt.resize.w );
         msg->addParam( (int64) evt.resize.h );
      break;

      case SDL_VIDEOEXPOSE:
         if ( vm->getSlot( "sdl_Expose", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_Expose" );
      break;

      case SDL_QUIT:
          if ( vm->getSlot( "sdl_Quit", false ) == 0 )
            return;

         msg = new VMMessage( "sdl_Quit" );
      break;

      default:
         params = 0;
   }

   vm->postMessage( msg );
}

/*#
   @method PollEvent SDL
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

FALCON_FUNC sdl_PollEvent( VMachine *vm )
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
   @method WaitEvent SDL
   @brief Waits forever until an event is received.

   This method blocks the current coroutine until a SDL event is
   received. However, the VM is able to proceed with other
   coroutines.

   As soon as a message is received and processed, the function returns, so
   it is possible to set a global flag in the message processors to communicate
   new program status to the subsequent code.

   In example, the following is a minimal responsive SDL Falcon application.

   @code
      object handler
         shouldQuit = false

         function on_sdl_Quit()
            self.shouldQuit = true
         end
      end

      ...
      ...
      
      // main code
      subscribe( "sdl_Quit", handler )
      
      while not handler.shouldQuit
         SDL.WaitEvent()
      end
   @endcode
   
   @note You can also start an automatic event listener and dispatcher in
   a parallel thread with @a SDL.StartEvents. This will oviate the need for
   a polling or waiting loop.
*/
bool sdl_WaitEvent_next( VMachine *vm )
{
   SDL_Event evt;

   int res = SDL_PollEvent( &evt );
   if ( res == 1 )
   {
      vm->returnHandler( 0 );  // do not call us anymore

      internal_dispatchEvent( vm, evt );
      // we're done -- but we have still a call pending
      return true;
   }
   else {
      // prepare to try again after a yield
      vm->yield( 0.01 );
      return true;
   }
}

FALCON_FUNC sdl_WaitEvent( VMachine *vm )
{
   SDL_Event evt;
   int res = SDL_PollEvent( &evt );
   if ( res == 1 )
   {
      internal_dispatchEvent( vm, evt );
   }
   else {
      // prepare to try again after a yield
      vm->returnHandler( sdl_WaitEvent_next );
      vm->yield( 0.01 );
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

   This method queries and/or changes the current processing
   status of an event. Disabling an event means that it won't be
   notified to event handlers anymore, and it will be silently
   discarded.

   It will still be possible to see the effect of that event by
   querying directly the event interface.

   State can be set to one of the following:

      - SDL.IGNORE: indicated event type will be automatically dropped from the event queue and will not be filtered.
      - SDL_ENABLE: indicated event type will be processed normally.
      - SDL_QUERY: SDL_EventState will return the current processing state of the specified event type.
*/
FALCON_FUNC sdl_EventState( VMachine *vm )
{
   Item *i_type;
   Item *i_state;

   if( vm->paramCount() != 2 ||
      ! ( i_type = vm->param(0) )->isOrdinal() ||
      ! ( i_state = vm->param(1) )->isOrdinal()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N" ) ) ;
      return;
   }

   vm->retval( (int64)
      ::SDL_EventState( (Uint8) i_type->forceInteger(), (int) i_state->forceInteger() ) );
}

/*#
   @method GetKeyState SDL
   @brief Gets a memory buffer which reflects current keyboard scancodes status.
   @return A 1 byte memory buffer.

   This method returns a memory buffer in which each element represents the status
   of a key scan code. The index of the code is provided by the @a SDLK enumeration;
   if a given item in the membuf is 1, then the key is currently pressed, else it
   it is released.

   This memory buffer is automaitcally updated when @a SDL.WaitEvent or
   @a SDL.PollEvent are called. If the program doesn't call those function,
   the @a SDL.PumpEvents method can be used to update current keyboard status.

   @note Calling this method more than once per program will cause cause useless duplication
   of the memory buffer.
*/

FALCON_FUNC sdl_GetKeyState( VMachine *vm )
{
   Uint8 *data;
   int size;

   data = ::SDL_GetKeyState( &size );
   // the data is static, needs no destructor.
   vm->retval( new MemBuf_1( data, size, 0 ) );
}

/*#
   @method GetModState SDL
   @brief Gets current keyboard modifier state.
   @return An integer containing or'd modifier state.

   The returned integer is a bitfield where active modifiers bits are
   turned on. The values are those listed by the @a SDLKMOD enumeration.
*/

FALCON_FUNC sdl_GetModState( VMachine *vm )
{
   vm->retval( (int64) ::SDL_GetModState() );
}

/*#
   @method SetModState SDL
   @brief Sets current keyboard modifier state.
   @param state the state to be set.

   This method will alter the keyboard modifier state for the application.
   The state parameter can be a or'd combination of @a SDLKMOD enumeration elements.
*/

FALCON_FUNC sdl_SetModState( VMachine *vm )
{
   Item *i_state;

   if( vm->paramCount() < 1 ||
      ! ( i_state = vm->param(0) )->isOrdinal()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) ;
      return;
   }

   ::SDL_SetModState( (SDLMod) i_state->forceInteger() );
}

/*#
   @method GetKeyName SDL
   @brief Gets a SDL specific name for a ceratin key
   @param key An @a SDLK value.
   @return a string containing the key name.
*/
FALCON_FUNC sdl_GetKeyName( VMachine *vm )
{
   Item *i_key;

   if( vm->paramCount() < 1 ||
      ! ( i_key = vm->param(0) )->isOrdinal()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) ;
      return;
   }

   vm->retval(
      new CoreString( ::SDL_GetKeyName( (SDLKey) i_key->forceInteger() ) ) );
}

/*#
   @method EnableUNICODE SDL
   @brief Enable or disable translation from keys to Unicode characters.
   @param mode Wether to enable, disable or query Unicode translation.
   @return current status of unicode translation.

   The parameter can be:
      - 1: enable unicode translation
      - 0: Disable unicode translation
      - -1: Query current unicode status.

   Falcon tunrs unicode translation ON at initialization, as characters
   in Falcon as treated as Unicode values, but it is possible to turn
   this off for better performance if character values are not needed.

*/
FALCON_FUNC sdl_EnableUNICODE( VMachine *vm )
{
   Item *i_mode;

   if( vm->paramCount() < 1 ||
      ! ( i_mode = vm->param(0) )->isInteger()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "I" ) ) ;
      return;
   }

   vm->retval(
      (int64) ::SDL_EnableUNICODE( (SDLKey) i_mode->forceInteger() ) );
}

/*#
   @method EnableKeyRepeat SDL
   @brief Enable or disable key repeat and set key rate.
   @param delay Delay before starting producing repeated keystores.
   @param interval Interval between keystores.
   @raise SDLError if requested parameters cannot be set.

   Enables or disables the keyboard repeat rate. delay specifies how long the key
   must be pressed before it begins repeating, it then repeats at the speed specified by interval.
   Both delay and interval are expressed in milliseconds.

   Setting delay to 0 disables key repeating completely. Good default values are
   SDL.DEFAULT_REPEAT_DELAY and SDL.DEFAULT_REPEAT_INTERVAL.

*/
FALCON_FUNC sdl_EnableKeyRepeat( VMachine *vm )
{
   Item *i_delay;
   Item *i_interval;

   if( vm->paramCount() < 2 ||
      ! ( i_delay = vm->param(0) )->isNumeric() ||
      ! ( i_interval = vm->param(1) )->isNumeric()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N" ) ) ;
      return;
   }

   if ( ::SDL_EnableKeyRepeat( (int) i_delay->forceInteger(), (int) i_interval->forceInteger() ) != 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 12, __LINE__ )
         .desc( "SDL Enable Key Repeat" )
         .extra( SDL_GetError() ) ) ;
   }
}

/*#
   @method GetMouseState SDL
   @brief Retreive mouse state
   @param MouseBuf
   @param interval Interval between keystores.
   @raise SDLError if requested parameters cannot be set.

   Enables or disables the keyboard repeat rate. delay specifies how long the key
   must be pressed before it begins repeating, it then repeats at the speed specified by interval.
   Both delay and interval are expressed in milliseconds.

   Setting delay to 0 disables key repeating completely. Good default values are
   SDL.DEFAULT_REPEAT_DELAY and SDL.DEFAULT_REPEAT_INTERVAL.

*/
FALCON_FUNC sdl_GetMouseState( VMachine *vm )
{
   Item *i_delay;
   Item *i_interval;

   if( vm->paramCount() < 2 ||
      ! ( i_delay = vm->param(0) )->isNumeric() ||
      ! ( i_interval = vm->param(1) )->isNumeric()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N,N" ) ) ;
      return;
   }

   if ( ::SDL_EnableKeyRepeat( (int) i_delay->forceInteger(), (int) i_interval->forceInteger() ) != 0 )
   {
      throw new SDLError( ErrorParam( FALCON_SDL_ERROR_BASE + 12, __LINE__ )
         .desc( "SDL Enable Key Repeat" )
         .extra( SDL_GetError() ) ) ;
   }
}


/*

SDL_GetMouseState -- Retrieve the current state of the mouse
SDL_GetRelativeMouseState -- Retrieve the current state of the mouse
SDL_GetAppState -- Get the state of the application
SDL_JoystickEventState -- Enable/disable joystick event polling
*/

/*#
   @method PumpEvents SDL
   @brief Update event listing and queueing during long operations.

   Normally, a responsive SDL-Falcon program should issue a @a SDL.WaitEvent
   loop in its main code to start marshalling events to listeners, but it is also possible
   to periodically poll the status of keyboard and other devices with direct query functions.

   To ensure the status is updated even when not calling WaitEvents, this method can be called.
   This will update all the internal device status representation, that will be then accurate
   if queried soon after.
*/

FALCON_FUNC sdl_PumpEvents( VMachine *vm )
{
   ::SDL_PumpEvents();
}

/*#
   @method GetAppState SDL
   @brief Gets current application state.
   @return Current application state

   This function may return one of the following values:

   - SDL_APPMOUSEFOCUS:  The application has mouse focus.
   - SDL_APPINPUTFOCUS:  The application has keyboard focus
   - SDL_APPACTIVE:  The application is visible
*/

FALCON_FUNC sdl_GetAppState( VMachine *vm )
{
   vm->retval( (int64) ::SDL_GetAppState() );
}

/*#
   @method JoystickEventState SDL
   @brief Changes joystick event propagation settings.
   @param mode Wether to disable, enable or query the joystick.
   @return Current joystick event propagation segging.

   This function is used to enable or disable joystick event processing.
   With joystick event processing disabled you will have to update joystick states
   with @a SDL.JoystickUpdate and read the joystick information manually.
   state is either SDL.QUERY, SDL.ENABLE or SDL.IGNORE.

   @note Joystick event handling is prefered

*/

FALCON_FUNC sdl_JoystickEventState( VMachine *vm )
{
   Item *i_code;

   if( vm->paramCount() < 1 ||
      ! ( i_code = vm->param(0) )->isInteger()
      )
   {
      throw new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "I" ) ) ;
      return;
   }

   vm->retval( (int64) ::SDL_JoystickEventState( (int) i_code->asInteger() ) );
}

/*#
   @method JoystickUpdate SDL
   @brief Updates the state(position, buttons, etc.) of all open joysticks.

   If joystick
   events have been enabled with SDL_JoystickEventState then this is called
   automatically in the event loop.
*/

FALCON_FUNC sdl_JoystickUpdate( VMachine *vm )
{
   ::SDL_JoystickUpdate();
}



//===================================================================
// Class SDLMouseState
//

FALCON_FUNC SDLMouseState_init( VMachine *vm )
{
   Inst_SdlMouseState* inst = dyncast<Inst_SdlMouseState*>( vm->self().asObject() );
   inst->setUserData( &inst->m_ms );
}

/*#
   @method Refresh SDLMouseState
   @brief Refresh current mouse position and status.

   Use this method only if the program pumps events
   or waits for them regularly. Otherwise, use
   @a SDLMouseState.PumpAndRefresh.

   @see SDL.WaitEvent
   @see SDL.PumpEvents
   @see SDL.StartEvents
*/
FALCON_FUNC SDLMouseState_Refresh( VMachine *vm )
{
   Inst_SdlMouseState* inst = dyncast<Inst_SdlMouseState*>( vm->self().asObject() );

   inst->m_ms.state = ::SDL_GetMouseState( &inst->m_ms.x, &inst->m_ms.y );
   ::SDL_GetRelativeMouseState( &inst->m_ms.xrel, &inst->m_ms.yrel );
}

/*#
   @method PumpAndRefresh SDLMouseState
   @brief Peeks incoming events into SDL and then refresh current mouse position and status.

   This method internally performs a @a SDL.PumpEvents call before
   calling the refresh method.
*/
FALCON_FUNC SDLMouseState_PumpAndRefresh( VMachine *vm )
{
   ::SDL_PumpEvents();
   SDLMouseState_Refresh( vm );
}


//===============================================================
// The event listener.
//
static bool s_bCoroTerminate = false;

bool _coroutinePollNext( VMachine *vm )
{
   SDL_Event evt;

   while( (!s_bCoroTerminate) && SDL_PollEvent( &evt )  )
   {
      internal_dispatchEvent( vm, evt );
   }

   if ( s_bCoroTerminate )
   {
      vm->returnHandler( 0 );
      s_bCoroTerminate = false;
      return false;
   }

   vm->yield( 0.05 );

   return true;
}


FALCON_FUNC _coroutinePoll( VMachine *vm )
{
   vm->returnHandler( _coroutinePollNext );
}


/*#
   @method StartEvents SDL
   @brief Stats dispatching of SDL events to this VM.

   This automatically starts an event listen loop on this virtual machine.
   Events are routed through the standard broadcasts.

   If a previous event listener thread, eventually on a different virtual machine,
   was active, it is stopped.
   
   Events are generated as broadcast message that can be received via 
   standard subscribe.

   Here follows the list of generated events.
  
   @section sdl_Active Event sdl_Active
	
	Application visibility event handler.
	
	Parameters of the event:
   - gain: 0 if the event is a loss or 1 if it is a gain.
   - state: SDL.APPMOUSEFOCUS if mouse focus was gained
         or lost, SDL.APPINPUTFOCUS if input focus was gained or lost,
         or SDL.APPACTIVE if the application was iconified (gain=0) or
         restored (gain=1).

   
   
   See SDL_ActiveEvent description in SDL documentation.

   Subscribe to this message method to receive ActiveEvent notifications.
   
   @section sdl_KeyDown Event sdl_KeyDown
   
   Keyboard key down event handler. Parameters generated:
   
   - state: SDL.PRESSED or SDL.RELEASED
   - scancode:  Hardware specific scancode
   - sym:  SDL virtual keysym
   - mod:  Current key modifiers
   - unicode:  Translated character

   See SDL_KeyboardEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_KEYDOWN notifications.

   @section sdl_KeyUp Event sdl_KeyUp
   
   Keyboard key up event handler. Parameters:
   
   - state: SDL.PRESSED or SDL.RELEASED
   - scancode:  Hardware specific scancode
   - sym:  SDL virtual keysym
   - mod:  Current key modifiers
   - unicode:  Translated character

   See SDL_KeyboardEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_KEYUP notifications.

   @section sdl_MouseMotion Event sdl_MouseMotion
   Mouse motion event handler. Parameters:
   
   - state: The current button state
   - x:  X coordinate of the mouse
   - y:  X coordinate of the mouse
   - xrel: relative movement of mouse on the X axis with respect to last notification.
   - yrel: relative movement of mouse on the X axis with respect to last notification.

   See SDL_MouseMotionEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_MOUSEMOTION notifications.

   @section sdl_MouseButtonDown Event sdl_MouseButtonDown
   Mouse button event handler. Parameters:
   - state: The current button state
   - button: The mouse button index (SDL_BUTTON_LEFT, SDL_BUTTON_MIDDLE, SDL_BUTTON_RIGHT)
   - x: X coordinate of the mouse
   - y: X coordinate of the mouse

   See SDL_MouseButtonEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_MOUSEBUTTONDOWN notifications.
   
   @section sdl_MouseButtonUp Event sdl_MouseButtonUp
   
   Mouse button event handler. Generated parameters:
   
   - state: The current button state
   - button: The mouse button index (SDL_BUTTON_LEFT, SDL_BUTTON_MIDDLE, SDL_BUTTON_RIGHT)
   - x: X coordinate of the mouse
   - y: X coordinate of the mouse

   See SDL_MouseButtonEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_MOUSEBUTTONUP notifications.
   
   @section sdl_JoyAxisMotion Event sdl_JoyAxisMotion

   Joystick axis motion event handler. Parameters:
   
   - which: Joystick device index
   - axis:  Joystick axis index
   - value:  Axis value (range: -32768 to 32767)

   See SDL_JoyAxisEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_JOYAXISMOTION notifications.
   
   @section sdl_JoyButtonDown Event sdl_JoyButtonDown
   
   Joystick button event handler. Parameters
   
   - which:  Joystick device index
   - button:  Joystick button index
   - state:  SDL_PRESSED or SDL_RELEASED

   See SDL_JoyButtonEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_JOYBUTTONDOWN notifications.
   
   @section sdl_JoyButtonUp Event sdl_JoyButtonUp
   
   Joystick button event handler
   
   - which:  Joystick device index
   - button:  Joystick button index
   - state:  SDL_PRESSED or SDL_RELEASED

   See SDL_JoyButtonEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_JOYBUTTONUP notifications.

   @section sdl_JoyHatMotion Event sdl_JoyHatMotion
   
   Joystick hat position change event handler. Parameters:
   
   - which:  Joystick device index
   - hat:  Joystick hat index
   - value:  hat position.

   See SDL_JoyHatEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_JOYHATMOTION notifications.

   @section sdl_JoyBallMotion Event sdl_JoyBallMotion
   
   Joystick trackball motion event handler. Parameters:
   
   - which:  Joystick device index
   - ball:  Joystick trackball index
   - xrel:  The relative motion in the X direction
   - yrel:  The relative motion in the Y direction

   See SDL_JoyBallEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_JOYBALLMOTION notifications.

   @section sdl_Resize Event sdl_Resize
   Window resize event handler. Parameters:
   
   - w:  New width of the window
   - h:  New height of the window

   See SDL_ResizeEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_VIDEORESIZE notifications.

   @section sdl_Expose Event sdl_Expose
   
   Window exposition (need redraw) notification. This event doesn't generate any
   parameter.

   See SDL_ExposeEvent description in SDL documentation.

   Subscribe to this message method to receive SDL_VIDEOEXPOSE notifications.

   
   @section sdl_Quit Event sdl_Quit
   
   Quit requested event.

   See SDL_Quit description in SDL documentation.
   
   Subscribe to this message method to receive SDL_Quit events.
   
   This notification means that the user asked the application to terminate.
   The application should call exit(0) if no other cleanup routines are needed,
   or perform cleanup and notifications for clean exit to other threads/coroutines.
*/

FALCON_FUNC sdl_StartEvents( VMachine *vm )
{
#ifdef USE_SDL_EVENT_THREADS
   s_mtx_events->lock();
   if ( s_EvtListener != 0 )
   {
      s_EvtListener->stop();
      delete s_EvtListener;
   }
   s_EvtListener = new SDLEventListener( vm );
   s_EvtListener->start();
   s_mtx_events->unlock();
#else
   Item* coro = vm->findWKI( "_coroutinePoll" );
   fassert( coro != 0 );
   vm->callCoroFrame( *coro, 0 );
#endif
}


/*#
   @method StopEvents SDL
   @brief Stops dispatching of SDL events.

   This immediately stops dispatching events.
   It is NOT necessary to perform this call before closing a program, but a
   VM may want to start manage events on its own, or to ignore them.

   If asynchronous event dispatching wasn't active, this call has no effect.
*/
FALCON_FUNC sdl_StopEvents( VMachine *vm )
{
#ifdef USE_SDL_EVENT_THREADS
   s_mtx_events->lock();
   if ( s_EvtListener != 0 )
   {
      s_EvtListener->stop();
      delete s_EvtListener;
      s_EvtListener = 0;
   }
   s_mtx_events->unlock();
#else
   s_bCoroTerminate = true;
#endif
}


FALCON_SERVICE SDLEventListener::SDLEventListener( VMachine* vm ):
   m_vm( vm ),
   m_th( 0 )
{
   vm->incref();
}

FALCON_SERVICE SDLEventListener::~SDLEventListener()
{
   m_vm->decref();
}

FALCON_SERVICE void* SDLEventListener::run()
{
   SDL_Event evt;

   while( ! m_eTerminated.wait(20) )
   {
      while( SDL_PollEvent( &evt ) )
      {
         //printf( "Dispatching event\n" );
         internal_dispatchEvent( m_vm, evt );
      }
   }

   return 0;
}

void FALCON_SERVICE SDLEventListener::start()
{
   if ( m_th == 0 )
   {
      m_th = new SysThread( this );
      m_th->start();
   }
}

void FALCON_SERVICE SDLEventListener::stop()
{
   if ( m_th != 0 )
   {
      m_eTerminated.set();
      void *dummy;
      m_th->join( dummy );
      m_th = 0;
   }
}

}
}
