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
      cs->decref();
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
   @optparam name The name of the mesasge managed by this VMSlot.

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

   If the @b name parameter is not given, the slot will be created as
   "anonymous", and won't be registered with this virtual machine. It will
   be possible to use it only through its methods.

   @section ev_anonym_slots Anonymous slots

   An anonymous slot can be created using an empty call to the VMSlot class
   constructor. This will make the slot private for the users that can
   access it; in other words, the slot won't be published to the VM and
   it won't be possible to broadcast on it using the standard functions
   @a broadcast or @a assert functions.

   @section ev_slots Automatic broadcast marshaling

   If a listener subscribed to a slot is a callable item, then it's called in
   case of broadcasts. If it's a non-callable property provider (class instances,
   blessed dictionaries, non-callable arrays) then a callable property named
   like "on_<slot_name>" is searched; if found, it's called (as a method of the
   host entity), otherwise a catch all method named "__on_event" is searched.

   If the "__on_event" method is found, it's called with the first parameter
   containing the broadcast name. Otherwise, an access error is raised.

   @section ev_events Events

   Events are "named broadcasts". They are issued on slots and excite all the listeners
   that are subscribed to that slot. The main difference is that automatic marshalling
   is directed to the name of the @i event rather to the name of the @i slot.

   See the following code:

   @code
      object Receiver
         _count = 0

         function display(): > "Count is now ", self._count
         function on_add(): self._count++
         function on_sub(): self._count--
         function __on_event( evname ): > "Received an unknown event: ", evname
      end

      s = VMSlot()  // creates an anonymous slot

      s.subscribe( Receiver )

      s.send( "add" ) // Instead of sending a broadcast ...
      s.send( "add" ) // ... generate some events via send()

      s.send( "A strange event", "some param" )  // will complain
      Receiver.display()   // show the count
   @endcode

   The @a VSMlot.send  method works similarly to the @a VMSlot.broadcast method,
   but it allows to specify an arbitrary event name. Callbacks subscribed to this
   slot would be called for @i every event, be it generated through a broadcast or
   via a send call.

   @section ev_subslots Registering to events and Sub-slots

   While callbacks subscribed to the slot will be excited no matter what kind of
   event is generated, it is possible to @i register callbacks to respond only to
   @i particular @i events via the @a VMSlot.register method.

   See the following example:

   @code
   slot = VMSlot()  // creates an anonymous slot
   slot.register( "first",
      { param => printl( "First called with ", param ) } )

   slot.register( "second",
      { param => printl( "Second called with ", param ) } )

   // send "first" and "second" events
   slot.send( "first", "A parameter" )
   slot.send( "second", "Another parameter" )

   // this will actually do nothing
   slot.broadcast( "A third parameter" )
   @endcode

   As no callback is @i subscribed to the slot, but some are just @i register to
   some events, a generic @i broadcast will have no effect.

   An interesting thing about registering to events is that a slot keeps tracks
   of callbacks and items registered to a specific event via a named slot. For
   example, to know who is currently subscribed to the "first" event, it's possible
   to call the @a VMSlot.getEvent method and inspect the returned slot. Any change
   in that returned slot will cause the event registration to change.

   For example, continuing the above code...

   @code
      //...
      fevt = slot.getEvent( "first" )

      // display each subscribed item
      for elem in fevt: > elem.toString()

      // and broadcast on the event first
      fevt.broadcast( "The parameter" )
   @endcode

   As seen, a broadcast on a sub-slot is equivalent to an event send on the parent
   slot.

   @note It is then possible to cache repeatedly broadcast slots, so that the virtual
   machine is not forced to search across the subscribed events.

   This structure can be freely replicated at any level. In the above example,
   @b fevt may be subject of send() and register() method application, and its own
   events can be retrieved trough its @a VMSlot.getEvent method.
*/

/*#
   @endgroup
*/

FALCON_FUNC VMSlot_init( ::Falcon::VMachine *vm )
{
   Item *i_msg = vm->param( 0 );

   if ( i_msg != 0 && ! (i_msg->isString()||i_msg->isNil()) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "[S]" ) );
   }

   CoreSlot* vms;
   if ( i_msg != 0 && i_msg->isString() )
   {
      vms = vm->getSlot( *i_msg->asString(), true );
      fassert( vms != 0 );
   }
   else
   {
      vms = new CoreSlot("");
   }

   CoreSlotCarrier* self = dyncast<CoreSlotCarrier *>( vm->self().asObjectSafe() );
   self->setSlot( vms );
   vms->decref();
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
   @method send VMSlot
   @brief Performs an event generation on this slot.
   @param event Event name.
   @param ... Extra parameters to be sent to listeners.

   The send method works as broadcast, with two major differences;
   - In case of objects being subscribed to this slot, the name of the
     broadcast message is that specified as a parameter, and not that
     of this slot. This means that automatic marshaling will be performed
     on methods named like on_<event>, and not on_<self.name>.
   - Items registered with @a VMSlot.register gets activated only if the
     @b event name is the same to which they are registered to.

*/

FALCON_FUNC VMSlot_send( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();

   Item *i_msg = vm->param( 0 );
   if ( i_msg == 0 || ! i_msg->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
   }

   // better to cache the name
   Item iName = *i_msg;
   cs->prepareBroadcast( vm->currentContext(), 1, vm->paramCount()-1, 0, iName.asString() );
}

/*#
   @method register VMSlot
   @brief Registers a listener for a specific event.
   @param event A string representing the event name.
   @param handler Handler to associate to this event.

   This function associates the given @b handler to a sub-slot named after
   the @b event parameter. This operation is equivalent to call
   @a VMSlot.getEvent() to create the desired sub-slot, and then call
   @a VMSlot.subscribe() on that named slot.

   @see VMSlot.send
*/

FALCON_FUNC VMSlot_register( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();

   Item *i_event = vm->param( 0 );
   Item *i_handler = vm->param( 1 );

   if ( i_event == 0 || ! i_event->isString()
       || i_handler == 0 || ! (i_handler->isComposed() || i_handler->isCallable() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S,C|O" ) );
   }

   // Get the child core slot
   CoreSlot* child = cs->getChild( *i_event->asString(), true );
   child->append( *i_handler );
}

/*#
   @method getEvent VMSlot
   @brief Returns an event (as a child VMSlot) handled by this slot.
   @param event A string representing the event name.
   @optparam force Pass true to create the event if it is not existing.
   @return a VMSlot representing the given event in this slot, or @b nil
           if not found.

   This method returns a named VMSlot that will be excited by @a VMSlot.send
   applied on this slot with the same @b event name.

   In other words, subscribing or unsubscribing items from the returned slot would
   add or remove listeners for a @a VMSlot.send call on this slot.

   Also, a broadcast on the returned VMSlot has the same effect of a @a VMSlot.send
   with the same name as the @b event passed.

   @see VMSlot.send
*/

FALCON_FUNC VMSlot_getEvent( ::Falcon::VMachine *vm )
{
   CoreSlot* cs = (CoreSlot*) vm->self().asObject()->getUserData();

   Item *i_event = vm->param( 0 );
   Item *i_force = vm->param( 1 );

   if ( i_event == 0 || ! i_event->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) );
   }

   // Get the child core slot
   CoreSlot* child = cs->getChild( *i_event->asString(),
         i_force != 0 && i_force->isTrue() ); // force creation if required
   if( child != 0 )
   {
      // found or to be created.
      Item* icc = vm->findWKI("VMSlot");
      fassert( icc != 0 );
      fassert( icc->isClass() );
      vm->retval( icc->asClass()->createInstance(child) );
      child->decref();
   }
   else
   {
      vm->retnil();
   }
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
