/*
   FALCON - The Falcon Programming Language
   FILE: indirect.cpp

   Indirect function calling and symbol access.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio apr 13 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include "core_messages.h"

/** \file
   Indirect function calling and symbol access.
*/

/*#
   @funset rtl_indirect Indirect call support
   @brief Functions meant to provide indirect symbol call facilities.

   @beginmodule core_module
   @beginset core_indirect
*/

namespace Falcon {
namespace core {

/*#
   @function call
   @brief Indirectly calls an item.
   @param callable A callable item
   @optparam parameters An array containing optional parameters.
   @return The return value of the indirectly called item.

   This function is meant to indirectly call a function. While pass
   and pass/in statements are meant to provide maximum efficiency in
   intercepting the functions, @b call is meant to provide different
   calling patterns (i.e. parameter mangling) in runtime.

   Using this function is equivalent to use a callable array, but it spares
   the need to add the callable item in front of the array containing the
   parameters.

   The following calls are equivalent:
   @code
      call( func, [1,2,3] )
      func( 1, 2, 3 )
      .[func 1 2 3]()
   @endcode

   The callable item may be any callable Falcon item. This includes normal
   functions,
   external functions, methods, classes (in which case their constructor
   is called and a new instance is returned) and Sigmas (callable sequences).

   The function returns the value that is returned by the called item.

   If the symbol that is in the position of the item to be called is not a
   callable object,
   a ParamError is raised.
*/

FALCON_FUNC  call( ::Falcon::VMachine *vm )
{
   Item *func_x = vm->param(0);
   Item *params_x = vm->param(1);

   if ( func_x == 0 || ! func_x->isCallable() ||
       ( params_x != 0 && ! params_x->isArray() ) )
       {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( "C,A" ) ) );
      return;
   }

   // fetch the item here, as we're going to change the vector.
   Item func = *func_x;

   uint32 count = 0;

   if ( params_x != 0 )
   {
      CoreArray *array = params_x->asArray();
      count = array->length();
      for( uint32 i = 0; i < count; i++ )
      {
         vm->pushParameter( (*array)[ i ] );
      }
   }

   vm->callFrame( func, count );
}

/*#
   @function methodCall
   @brief Indirectly calls an object method.
   @param object An object whose method is to be called.
   @param methodName The name of a method to be called.
   @optparam parameters An array containing optional parameters.
   @return The return value of the called method.
   @raise AccessError if the object doesn't contain desired property
   @raise TypeError if the property exists, but is not callable.

   This function is meant to indirectly call a method whose name is known.
   It's effect it's the same as calling @a getProperty to retrieve the
   method object and then using that item in the @a call function; however,
   the item is not repeatedly needed, this function is more efficient.

   @b methodCall returns the same value that is returned by the method.
   If the object does not contain the required property, an AccessError is
raised.
   If the property exists but it's not a callable item, a TypeError is raised.
*/

FALCON_FUNC  methodCall( ::Falcon::VMachine *vm )
{
   Item *obj_x = vm->param(0);
   Item *method_x = vm->param(1);
   Item *params_x = vm->param(2);

   if ( obj_x == 0 || ! obj_x->isObject() || method_x == 0 || ! method_x->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
            //"requres an object and a string representing a method" );
      return;
   }

   if ( params_x != 0 && ! params_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         //"optional third parameter must be an array" );
      return;
   }

   Item method;
   CoreObject *self = obj_x->asObject();
   if ( ! self->getProperty( *method_x->asString(), method ) ) {
      vm->raiseModError( new AccessError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( ! method.isCallable() ) {
      vm->raiseModError( new TypeError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int count = 0;
   if ( params_x != 0 )
   {
      CoreArray *array = params_x->asArray();
      Item *elements = array->elements();
      count = array->length();
      for ( int i = 0; i < count; i ++ ) {
         vm->pushParameter( elements[ i ] );
      }
   }

   method.methodize( self );
   vm->callFrame( method, count );
}


static void internal_marshal( VMachine *vm, Item *message, Item *prefix,
    Item *if_not_found,
   const char *func_format )
{
  if ( ! vm->sender().isObject() && ! vm->self().isObject() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString(rtl_sender_not_object ) ) ) );
      return;
   }

   if ( message == 0 ||  ! message->isArray() ||
         ( prefix != 0 && ! prefix->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( func_format ) ) );
      return;
   }

   CoreArray &amsg = *message->asArray();
   if( amsg.length() == 0 || ! amsg[0].isString() || amsg[0].asString()->size() == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_marshall_not_cb ) ) ) );
      return;
   }

   // Ok, we're clear.
   // should we add a prefix to the marshalled element?
   String *method_name;
   if ( prefix != 0 && prefix->asString()->size() > 0 )
   {
      method_name = new GarbageString( vm, *amsg[0].asString() );
      const String &temp = *prefix->asString();
      // is the last character a single quote?
      if ( temp.size() > 0 )
      {
         if( temp.getCharAt( temp.length() - 1 ) == '\'' )
         {
            uint32 cFront = method_name->getCharAt(0);
            if ( cFront >= 'a' && cFront <= 'z' )
               method_name->setCharAt( 0, cFront - 'a' + 'A' );

            method_name->prepend( temp.subString(0, temp.length() - 1 ) );
         }
         else
            method_name->prepend( temp );
      }
   }
   else
      method_name = amsg[0].asString();


   // do the marshalled method exist and is it callable?
   Item method;
   CoreObject *self = vm->self().isObject() ? vm->self().asObject() : vm->sender().asObject();
   if ( ! self->getProperty( *method_name, method ) ||
        ! method.isCallable() )
   {
      // if not, call the item
      if ( if_not_found == 0 )
      {
         vm->raiseModError( new AccessError( ErrorParam( e_non_callable, __LINE__ ).
            origin(e_orig_runtime) ) );
         return;
      }
      else
         vm->retval( *if_not_found );
      return;
   }

   int count = 0;
   count = amsg.length();
   for ( int i = 1; i < count; i ++ ) {
      vm->pushParameter( amsg[ i ] );
   }

   method.methodize( self );
   vm->callFrame( method, count-1 );
}

/*#
   @function marshalCB
   @brief Perform an event marshaling callback from inside an object.
   @param message The message to be marshalled.
   @optparam prefix An optional prefix to be applied to event handing method names.
   @optparam when_not_found Value to be returned when the event cannot be handled.
   @return The return value of the handler, or the value of @b when_not_found if
      the message couldn't be handled.
   @raise AccessError if the message couldn't be handled, and a default  @b when_not_found
      value is not given.

   This function performs automatic callback on objects that should receive and
   handle events.

   The event is an array whose first element is the event name, and the other
   elements are the parameters for the handler. This function will search the
   calling object for a callable property having the same name of the event
   that should be handled. MarshallCB may also search the current object, if it
   is directly assigned to a property of an object, and not called from an object.

   An optional prefix for the callback handlers may be given. If given, an
   handler   for a certain event should be prefixed with that string to manage a
   certain   event. In example, suppose that all the event handlers in the current
   object are   prefixed with “on_”, so that the event “new_item” is handled by
   “on_new_item”   handler, “item_removed” is handled by “on_item_removed” and so
   on. Then, it is   possible to pass “on_” as the prefix parameter.

   Some prefer to have handlers with a prefix and the first letter of the event
   capitalized. In example, it is common to have event “newFile” handled by
   “onNewFile” method, and so on. This convention will be acknowledged and used
   if   the last character of the handler prefix is a single quote; in this
   example, to   handle “newFile” (or “NewFile”) events through and “onNewFile”
   handler, the   prefix should be set as “on'”.

   The when_not_found parameter is returned in case the event handler is not
   found.   It is a common situation not to be willing to handle some events, and
   that is   not necessarily an error condition. The marshalCB function may return,
   in   example, an out of band item to signal this fact, or it may just
   return a reasonable “not processed” value, as nil or false. If this parameter
   is   not provided, not finding a valid handler will cause an access error to be
   raised.

   @see oob
*/

FALCON_FUNC  marshalCB( ::Falcon::VMachine *vm )
{
   Item *message = vm->param(0);
   Item *prefix = vm->param(1);
   Item *if_not_found = vm->param(2);

   internal_marshal( vm, message, prefix, if_not_found, "A,[S,X]" );
}

/*#
   @function marshalCBR
   @brief Perform an event marshaling callback from inside an object.
   @param prefix An optional prefix to be applied to event handing method names.
   @param message The message to be marshalled.
   @return The return value of the handler.
   @raise AccessError if the message couldn't be handled.

   This method works as @a marshalCB, but the order of the parameter is changed
   to make it easier to be used directly as object properties.

   As the marshalled message is added as extra parameter during the
   marshalling method call, it is possible to assign @b marshalCBR
   directly to object properties, asin this example:
   @code
   attributes: receiver

   object MyReceiver
      // automatic marshalling
      receiver = [marshalCBR, "on'"]

      // Message handler
      function onNewMessage( data )
         > "New Message: ", data
      end

      has receiver
   end

   broadcast( receiver, ["newMessage", "data sent as a message"] )
   @endcode

   Conversely to @a marshalCBX, marshalCBR will raise an access error
   in case the handler is not found.
*/

FALCON_FUNC  marshalCBX( ::Falcon::VMachine *vm )
{
   Item *prefix = vm->param(0);
   Item *if_not_found = vm->param(1);
   Item *message = vm->param(2);

   internal_marshal( vm, message, prefix, if_not_found, "S,X,A" );
}

/*#
   @function marshalCBX
   @brief Perform an event marshaling callback from inside an object.
   @param prefix An optional prefix to be applied to event handing method names.
   @param when_not_found The return value to be provided when the message
                         cannot be marshalled.
   @param message The message to be marshalled.
   @return The return value of the handler, or the value of @b when_not_found if
      the message couldn't be handled.

   This method works as marshalCB, but the order of the parameter is changed to
   make it easier to be used directly as object properties. Consider the following
   example:

   @code
      object handler
         marshaller = [marshalCBX, "on_", false ]

         function on_new_item( item )
            > "a new item: ", item
         end
      end

      event = ["new_item", 0 ]
      handler.marshaller( event )
   @endcode

   In this way it is possible to stuff the marshaller in a simple declaration
   inside the object property declaration block, instead defining a marshaling
   method that had to call marshalCBX.
*/
FALCON_FUNC  marshalCBR( ::Falcon::VMachine *vm )
{
   Item *prefix = vm->param(0);
   Item *message = vm->param(1);

   internal_marshal( vm, message, prefix, 0, "S,A" );
}

}
}

/* end of indirect.cpp */
