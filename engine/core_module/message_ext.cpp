/*
   FALCON - The Falcon Programming Language.
   FILE: message_ext.cpp

   Attribute support functions for scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 02:33:46 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"

namespace Falcon {
namespace core {


static void check_assertion( VMachine *vm, CoreSlot *cs, const Item &itm )
{
   if ( cs->hasAssert() )
   {
      vm->pushParameter( cs->assertion() );

      if ( ! itm.isCallable() )
      {
         if( itm.isComposed() )
         {
            // may throw
            Item tmp;
            itm.asDeepItem()->readProperty( "on_" + cs->name(), tmp );
            tmp.readyFrame( vm, 1 );
         }
         else
         {
            throw new CodeError( ErrorParam( e_non_callable, __LINE__ ).extra( "broadcast" ) );
         }
      }
      else
         itm.readyFrame( vm, 1 );
   }
}

/*#
   @group message_model Message oriented model
   @brief Functions and classes supporting Message Oriented Programming (MOP)

   @begingroup message_model
*/

/*#
   @class VMSlot
   @brief Subscription slot for Message Oriented Programming.
*/

/*#
   @function broadcast
   @param msg A message (string) to be broadcast.
   @param ... Zero or more data to be broadcaset.
   @brief Send a message to every object having an attribute.

   This function iterates over all the items having a certain attribute; if those objects provide a method
   named exactly as the attribute, then that method is called. A method can declare that it has "consumed"
   the message (i.e. done what is expected to be done) by returning true. In this case, the call chain is
   interrupted and broadcast returns. A method not wishing to prevent other methods to receive the incoming
   message must return false. Returning true means "yes, I have handled this message,
   no further processing is needed".

   It is also possible to have the caller of broadcast to receive a return value created by the handler;
   If the handler returns an out of band item (using @a oob) propagation of the message is stopped and
   the value is returned directly to the caller of the @b broadcast function.

   The order in which the objects receive the message is random; there isn't any priority across a single
   attribute message. For this reason, the second form of broadcast function is provided. To implement
   priority processing, it is possible to broadcast a sequence of attributes. In that case, all the
   objects having the first attribute will receive the message, and will have a chance to stop
   further processing, before any item having the second attribute is called and so on.

   The broadcast function can receive other parameters; in that case the remaining parameters are passed
   as-is to the handlers.

   Items having a certain attribute and receiving a broadcast over it need not to implement an handler.
   If they don't provide a method having the same name of the broadcast attribute, they are simply
   skipped. The same happens if they provide a property which is not callable; setting an handler to
   a non-callable item is a valid operation to disable temporarily message processing.

   An item may be called more than once in a single chained broadcast call if it has more than one of
   the attributes being broadcast and if it provides methods to handle the messages.

   It is possible to receive more than one broadcast in the same handler using the "same handler idiom":
   setting a property to a method of the same item in the init block or in the property initialization.
   In example:

   @code
      attributes: attr_one, attr_two

      object handler
         attr_two = attr_one
         function attr_one( param )
            // do something
            return false
         end
         has attr_one, attr_two
      end
   @endcode
*/

FALCON_FUNC  broadcast( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   if ( ! i_msg->isString() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S,..." ) ) );
      return;
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), false );
   if ( cs == false )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_range ).
         extra( "message not found" ) ) );
      return;
   }

   cs->prepareBroadcast( vm, 1, vm->paramCount() - 1 );
}

/*#
   @function subscribe
   @brief Registers a callback to a message slot.
   @param msg A string with the message name on which the item should be registered.
   @param handler A callable item or instance providing callback support.
*/

FALCON_FUNC subscribe( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   Item *i_handler = vm->param( 1 );
   if ( i_msg == 0 || ! i_msg->isString()
        || i_handler == 0  || ! ( i_handler->isCallable() || i_handler->isComposed() ) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S,C" ) ) );
      return;
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), true );
   cs->push_back( *i_handler );
   check_assertion( vm, cs, *i_handler );
}

/*#
   @function unsubscribe
   @brief Unregisters a registered callback from a slot.
   @param msg A string with the message name on which the item should be registered.
   @param handler A callable item or instance providing callback support.
   @raise CodeError if the @b handler is not registered with this slot.
   @raise AccessError if the named message slot doesn't exist.
*/

FALCON_FUNC unsubscribe( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   Item *i_handler = vm->param( 1 );

   if ( i_msg == 0 || ! i_msg->isString()
        || i_handler == 0 || ! ( i_handler->isCallable() || i_handler->isComposed() ) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S,C" ) ) );
      return;
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), false );
   if ( cs == 0 )
   {
      vm->raiseRTError( new AccessError( ErrorParam( e_inv_params ).
         extra( *i_msg->asString() ) ) );
      return;
   }

   if( ! cs->remove( *i_handler ) )
   {
      vm->raiseRTError( new CodeError( ErrorParam( e_param_range, __LINE__ ).extra( "unregister" ) ) );
   }

   if ( cs->empty() && ! cs->hasAssert() )
   {
      vm->removeSlot( cs->name() );
   }
}


/*#
   @function getSlot
   @brief Retreives a MOP Message slot.
   @param msg The message slot that must be taken or created.
   @param handler A callable item or instance providing callback support.
   @return The message slot coresponding with this name.
*/
FALCON_FUNC getSlot( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );

   if ( i_msg == 0 || ! i_msg->isString() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S" ) ) );
      return;
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString()  );

   Item* cc_slot = vm->findWKI( "VMSlot" );
   fassert( cc_slot != 0 );
   cs->incref();
   CoreObject *obj = cc_slot->asClass()->createInstance( cs );
   vm->retval( obj );
}


/*#
   @function consume
   @brief Consumes currently being broadcasted signal.
*/
FALCON_FUNC consume( ::Falcon::VMachine *vm )
{
   // TODO: report error.
   vm->consumeSignal();
}

/*#
   @function assert
   @brief Creates a message assertion on a certain message slot.
   @param msg The message to be asserted.
   @param data The value of the assertion.

   If there are already subscribed callbacks for this message
   a broadcast on them is performed now.
*/
FALCON_FUNC assert( ::Falcon::VMachine *vm )
{
   // TODO: report error.
   Item *i_msg = vm->param( 0 );
   Item *i_data = vm->param( 1 );
   if ( i_msg == 0 || ! i_msg->isString() || i_data == 0  )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S,X" ) ) );
      return;
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), true );
   cs->assert( vm, *i_data );
}

/*#
   @function retract
   @brief Removes a previous assertion on a message.
   @param msg The message slot to be retracted.
*/
FALCON_FUNC retract( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   if ( i_msg == 0 || ! i_msg->isString()  )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S" ) ) );
      return;
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), true );
   if( cs != 0 )
   {
      cs->retract();
      if( cs->empty() )
      {
         vm->removeSlot( cs->name() );
      }
   }

     // TODO: report error if the slot is not found
}

/*#
   @function getAssert
   @brief Returns the given assertion, if it exists.
   @param msg The message slot on which the assertion is to be ckeched.
   @optparam default If given, instead of raising in case the essartion is not found, return this item.
   @raise MessageError if the given message is not asserted.
*/
FALCON_FUNC getAssert( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   if ( i_msg == 0 || ! i_msg->isString()  )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S" ) ) );
      return;
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), true );
   if ( cs == 0 )
   {
      vm->raiseRTError( new MessageError( ErrorParam( e_inv_params ).
         extra( *i_msg->asString() ) ) );
      return;
   }
   else
   {
      if( cs->hasAssert() )
      {
         vm->regA() = cs->assertion();
      }
      else 
      {
         if ( vm->param(0) )
         {
            vm->regA() = *vm->param(0);
         }
         else {
            vm->raiseRTError( new MessageError( ErrorParam( e_inv_params ).
               extra( *i_msg->asString() ) ) );
         }
   
      }
   }
}

//================================================================================
// VMSlot wrapper class.
//


/*#
   @method broadcast VMSlot
   @brief Performs broadcast on this slot.
   @param ... Extra parameters to be sent to listeners.

*/

FALCON_FUNC VMSlot_broadcast( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   cs->prepareBroadcast( vm, 0, vm->paramCount() );
}

/*#
   @method subscribe VMSlot
   @brief Registers a callback handler on this slot.
   @param handler A callable item or instance providing callback support.
*/

FALCON_FUNC VMSlot_subscribe( ::Falcon::VMachine *vm )
{
   Item *callback = vm->param(0);
   if ( callback == 0 || ! ( callback->isCallable() || callback->isComposed() ) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "C" ) ) );
      return;
   }

   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   cs->push_back( *callback );
   check_assertion( vm, cs, *callback );
}

/*#
   @method unsubscribe VMSlot
   @brief Unregisters a callback handler from this slot.
   @param handler The callable that must be unregistered.
   @raise CodeError if the @b handler is not registered with this slot.
*/

FALCON_FUNC VMSlot_unsubscribe( ::Falcon::VMachine *vm )
{
   Item *callback = vm->param(0);
   if ( callback == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S,C" ) ) );
      return;
   }

   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   if( ! cs->remove( *callback ) )
   {
      vm->raiseRTError( new CodeError( ErrorParam( e_param_range, __LINE__ ).extra( "unregister" ) ) );
   }

   if ( cs->empty() && ! cs->hasAssert() )
   {
      vm->removeSlot( cs->name() );
   }
}

/*#
   @method prepend VMSlot
   @brief Registers a callback handler that will be called before the others.
   @param handler The callable that must be unregistered.
*/

FALCON_FUNC VMSlot_prepend( ::Falcon::VMachine *vm )
{
   Item *callback = vm->param(0);
   if ( callback == 0 || ! ( callback->isCallable() || callback->isComposed() ) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "C" ) ) );
      return;
   }

   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   cs->push_front( *callback );
   check_assertion( vm, cs, *callback );
}

/*#
   @method assert VMSlot
   @brief Creates a message assertion on this certain message slot.
   @param data The value of the assertion.

   If there are already subscribed callbacks for this message
   a broadcast on them is performed now.
*/
FALCON_FUNC VMSlot_assert( ::Falcon::VMachine *vm )
{
   Item *i_data = vm->param( 0 );
   if ( i_data == 0  )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "X" ) ) );
      return;
   }

   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   cs->assert( vm, *i_data );
}

/*#
   @method retract VMSlot
   @brief Removes a previous assertion on a message.
*/
FALCON_FUNC VMSlot_retract( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   cs->retract();
   if( cs->empty() )
   {
      vm->removeSlot( cs->name() );
   }
}

/*#
   @method getAssert VMSlot
   @brief Gets the item asserted for this slot.
   @optparam default If given, instead of raising in case the essartion is not found, return this item.
   @raise MessageError if the item has not an assertion and a default is not given.
*/
FALCON_FUNC VMSlot_getAssert( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   if( cs->hasAssert() )
   {
      vm->regA() = cs->assertion();
   }
   else
   {
      if ( vm->param(0) )
      {
         vm->regA() = *vm->param(0);
      }
      else {
         vm->raiseRTError( new MessageError( ErrorParam( e_inv_params ).
             extra( "..." ) ) );
      }
   }
}


}
}

/* end of message_ext.cpp */
