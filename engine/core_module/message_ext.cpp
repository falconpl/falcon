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
   @funset set_message_model Message oriented model functions
   @brief Functions supporting Message Oriented Programming (MOP)
   @see message_model

   This is the list of functions working together to implement the
   message oriented model. Other than this functions, it is possible
   to use the @a VMSlot class for direct access to reflected virtual
   machine slots. On this regards, see the @a message_model group.
*/

/*#
   @group message_model Message oriented model
   @brief Functions and classes supporting Message Oriented Programming (MOP)

   @begingroup message_model
*/


/*#
   @function broadcast
   @param msg A message (string) to be broadcast.
   @optparam ... Zero or more data to be broadcaset.
   @brief Sends a message to every callable item subscribed to a message.
   @return true if @b msg is found, false if it doesn't exist.
   @inset set_message_model
   @see VMSlot
   @see getSlot

   Broadcast function implicitly searches for a Virtual Machine Message Slot (@a VMSlot)
   with the given @b msg name, and if it finds it, it emits a broadcast on that.

   If the message is not found, the broadcast is silently dropped (no error is raised),
   but the function returns false.

   As calling this function requires a scan in the virtual machine message slot
   table, in case of repeated operations it is preferable to explicitly search for
   a slot with @a getSlot, or to create it as an @a VMSlot instance. On the other hand,
   if the reference to a VMSlot is not needed, this function allows to broadcast on the
   slot without adding the overhead required by the creation of the @a VMSlot wrapper.
*/

FALCON_FUNC  broadcast( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   if ( ! i_msg->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S,..." ) );
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), false );
   if ( cs )
   {
      cs->prepareBroadcast( vm->currentContext(), 1, vm->paramCount() - 1 );
      vm->regA().setBoolean(true);
   }
   else
      vm->regA().setBoolean(false);

   // otherwise, we're nothing to do here.
}

/*#
   @function subscribe
   @inset set_message_model
   @brief Registers a callback to a message slot.
   @param msg A string with the message name on which the item should be registered.
   @param handler A callable item or instance providing callback support.
   @optparam prio Set to true to insert this subscription in front of the subscription list.
*/

FALCON_FUNC subscribe( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   Item *i_handler = vm->param( 1 );
   Item *i_prio = vm->param(2);
   
   if ( i_msg == 0 || ! i_msg->isString()
        || i_handler == 0  || ! ( i_handler->isCallable() || i_handler->isComposed() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin( e_orig_runtime )
         .extra( "S,C" ) );
   }

   String *sub = i_msg->asString();
   CoreSlot* cs = vm->getSlot( *sub, true );
   if ( i_prio != 0 && i_prio->isTrue() )
      cs->push_front( *i_handler );
   else
      cs->push_back( *i_handler );
      
   check_assertion( vm, cs, *i_handler );
}

/*#
   @function unsubscribe
   @inset set_message_model
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
      throw new ParamError( ErrorParam( e_inv_params ).
         extra( "S,C" ) );
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), false );
   if ( cs == 0 )
   {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( *i_msg->asString() ) );
   }

   if( ! cs->remove( *i_handler ) )
   {
      throw new CodeError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime ).extra( "unsubscribe" ) );
   }

   if ( cs->empty() && ! cs->hasAssert() )
   {
      vm->removeSlot( cs->name() );
   }
}


/*#
   @function getSlot
   @inset set_message_model
   @brief Retreives a MOP Message slot.
   @param msg The message slot that must be taken or created.
   @optparam make If true (default) create the slot if it doesn't exist.
   @return The message slot coresponding with this name.
*/
FALCON_FUNC getSlot( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );

   if ( i_msg == 0 || ! i_msg->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), (vm->param(1) == 0 || vm->param(1)->isTrue())  );
   if ( cs == 0 )
   {
      throw new MessageError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( *i_msg->asString() ) );
   }
   else
   {
      Item* cc_slot = vm->findWKI( "VMSlot" );
      fassert( cc_slot != 0 );
      // the factory function takes care of increffing cs
      CoreObject *obj = cc_slot->asClass()->createInstance( cs );
      vm->retval( obj );
   }
}


/*#
   @function consume
   @inset set_message_model
   @brief Consumes currently being broadcasted signal.
*/
FALCON_FUNC consume( ::Falcon::VMachine *vm )
{
   // TODO: report error.
   vm->consumeSignal();
}

/*#
   @function assert
   @inset set_message_model
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S,X" ) );
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), true );
   cs->setAssertion( vm, *i_data );
}

/*#
   @function retract
   @inset set_message_model
   @brief Removes a previous assertion on a message.
   @param msg The message slot to be retracted.
*/
FALCON_FUNC retract( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   if ( i_msg == 0 || ! i_msg->isString()  )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
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
   @inset set_message_model
   @brief Returns the given assertion, if it exists.
   @param msg The message slot on which the assertion is to be ckeched.
   @optparam default If given, instead of raising in case the assertion is not found, return this item.
   @raise MessageError if the given message is not asserted.
*/
FALCON_FUNC getAssert( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );
   Item *i_defalut = vm->param(1);
   
   if ( i_msg == 0 || ! i_msg->isString()  )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
   }

   CoreSlot* cs = vm->getSlot( *i_msg->asString(), true );
   if ( cs == 0 )
   {
      if ( i_defalut != 0 )
      {
         vm->regA() = *i_defalut;
      }
      else {
         throw new MessageError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *i_msg->asString() ) );
      }
   }
   else
   {
      if( cs->hasAssert() )
      {
         vm->regA() = cs->assertion();
      }
      else
      {
         if ( i_defalut != 0 )
         {
            vm->regA() = *i_defalut;
         }
         else {
            throw new MessageError( ErrorParam( e_inv_params )
               .origin( e_orig_runtime )
               .extra( *i_msg->asString() ) );
         }

      }
   }
}

//================================================================================
// VMSlot wrapper class.
//

/*#
   @class VMSlot
   @brief VM Interface for message oriented programming operations.
   @param name The name of the mesasge managed by this VMSlot.

   The VMSlot instance is a direct interface to the messaging
   facility of the VM creating it. It is implicitly created by
   the @a getSlot function, but it can be directly created by
   the user.

   If a slot with the given name didn't previously exist,
   a new message slot is created in the virtual machine, otherwise
   the already existing slot is wrapped in the returned instance.

   @code
      // create a message slot
      x = VMSlot( "message" )
      x.subscribe( handler )
      ...
      y = VMSlot( "message" )
      y.broadcast( "value" )  // handler is called.
   @endcode

   Same happens if the VMSlot is created via @a getSlot, or implicitly
   referenced via @a subscribe function. Slots are considered
   unique by name, so that comparisons on slots are performed
   on their names.
*/

/*#
   @endgroup
*/

FALCON_FUNC VMSlot_init( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );

   if ( i_msg == 0 || ! i_msg->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
   }

   CoreSlot* vms = vm->getSlot( *i_msg->asString(), true );
   fassert( vms != 0 );

   CoreSlotCarrier* self = dyncast<CoreSlotCarrier *>( vm->self().asObjectSafe() );
   self->setSlot( vms );
}


/*#
   @method broadcast VMSlot
   @brief Performs broadcast on this slot.
   @param ... Extra parameters to be sent to listeners.

*/

FALCON_FUNC VMSlot_broadcast( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   cs->prepareBroadcast( vm->currentContext(), 0, vm->paramCount() );
}

/*#
   @method name VMSlot
   @brief Returns the name of this slot
   @return The name of the event bind to this slot (as a string).
*/

FALCON_FUNC VMSlot_name( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   vm->retval( new CoreString(cs->name()) );
}


/*#
   @method subscribe VMSlot
   @brief Registers a callback handler on this slot.
   @param handler A callable item or instance providing callback support.
   @optparam prio Set to true to have this handler called before the previous ones.
*/

FALCON_FUNC VMSlot_subscribe( ::Falcon::VMachine *vm )
{
   Item *callback = vm->param(0);
   Item *i_prio = vm->param(1);
   if ( callback == 0 || ! ( callback->isCallable() || callback->isComposed() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "C" ));
   }

   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   if( i_prio != 0 && i_prio->isTrue() )
      cs->push_front( *callback );
   else
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S,C" ) );
   }

   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   if( ! cs->remove( *callback ) )
   {
      throw new CodeError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "unregister" ) );
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "C" ) );
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "X" ) );
   }

   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   cs->setAssertion( vm, *i_data );
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
      if ( vm->paramCount() > 0 )
      {
         vm->regA() = *vm->param(0);
      }
      else {
         throw new MessageError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "..." ) );
      }
   }
}

/*#
   @method first VMSlot
   @brief Gets an iterator to the first subscriber.
   @return An iterator to the first subscriber of this message slot.
*/
FALCON_FUNC VMSlot_first( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   Item* cc = vm->findWKI( "Iterator" );
   fassert( cc != 0 );
   CoreObject *oi = cc->asClass()->createInstance( new Iterator( cs ) );
   vm->retval( oi );
}

/*#
   @method last VMSlot
   @brief Gets an iterator to the last subscriber.
   @return An iterator to the last subscriber of this message slot.
*/
FALCON_FUNC VMSlot_last( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();
   Item* cc = vm->findWKI( "Iterator" );
   fassert( cc != 0 );
   CoreObject *oi = cc->asClass()->createInstance( new Iterator( cs, true) );
   vm->retval( oi );
}
}
}

/* end of message_ext.cpp */
