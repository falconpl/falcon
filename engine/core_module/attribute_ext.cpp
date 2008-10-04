/*
   FALCON - The Falcon Programming Language.
   FILE: functional_ext.cpp

   Attribute support functions for scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 02:33:46 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/attribute.h>

namespace Falcon {
namespace core {

/*#
   @funset attrib_model Attribute model support
   @brief Functions supporting attributes.

   Attributes define dynamic boolean characteristics that instances may have at a certain moment.
   An attributed can be given or removed from a certain object, or automatically given to new
   instances through class declaration has statement. The VM keeps track of instances having attributes,
   so it is possible to iterate on, or send a message to, all the objects having a certain attribute.
   In this section, the functions that allow to access this functionalities are explained.

   Attribute can be treated as collections in for/in loops and iterators can be extracted from
   them with the first() BOM method and with the Iterator class constructor.
*/


/*#
   @function attributeByName
   @inset attrib_model
   @brief Returns an attribute registered with the VM as by symbolic name.
   @param name The attribute name that has been registered with the VM.
   @optParam raiseIfNotFound if given and true, the function will raise an error in case
             the attribute with the given name is not found.
   @return The attribute having the given name, if found, or nil.
   @raise AccessError if the attribute is not found and the raise is required.

   Attributes registered with the VM through a local export are made available in a special
   index which can be accessed by name. In example the \b give statement can refer symbols
   containing attributes or globally visible attribute names.

   This function returns an attribute which is globally visible. In this way it is possible
   to query the VM for attributes to be fed in the other core functions working with
   attributes.
*/
FALCON_FUNC  attributeByName( ::Falcon::VMachine *vm )
{
   Item *itm = vm->param( 0 );
   if ( ! itm->isString() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S" ) ) );
      return;
   }

   Attribute *attrib = vm->findAttribute( *itm->asString() );
   if ( attrib == 0 ) {
      if ( vm->param(1) != 0 && vm->param(1)->isTrue() )
      {
         vm->raiseRTError( new AccessError( ErrorParam( e_undef_sym ).
            extra( *vm->param(0)->asString() ) ) );
         return;
      }

      vm->retnil();
   }
   else {
      vm->retval( attrib );
   }
}


/*#
   @function having
   @inset attrib_model
   @brief Returns an array containing all the items having a certain attribute.
   @param attrib The attribute that will be inspected
   @return An array with all the items currently having that attribute.

   If the attribute isn't currently given to any item, this function will return an
   empty array. Notice that it is not strictly necessary to call having function
   just to iterate over the items i.e. in a for/in loop, as attributes
   can be directly iterated:

   @code
   attributes: opened
   //....
   for item in opened
      > "Item ", item.name, " has opened"
   end
   @endcode

   Also, attributes support iterators; an Iterator instance can be built passing an attribute
   as the parameter of the constructor, and first() BOM method can be called on an
   attribute to create an iterator which can be used to inspect all the objects having
   a certain attribute.

   However, having function is still useful when a snapshot of the items currently having
   a certain attribute is needed. In example, it is possible to save in a variable the array,
   change the status of some objects by removing the attribute from them and finally re-giving
   the attribute.
*/
FALCON_FUNC  having( ::Falcon::VMachine *vm )
{
   Item *itm = vm->param( 0 );
   if ( ! itm->isAttribute() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "a" ) ) );
      return;
   }

   Attribute *attrib = itm->asAttribute();
   AttribObjectHandler *head = attrib->head();
   CoreArray *arr = new CoreArray( vm );
   while( head != 0 )
   {
      arr->append( head->object() );
      head = head->next();
   }

   vm->retval( arr );
}

/*#
   @function testAttribute
   @inset attrib_model
   @brief Checks if an object is currently given a certain attribute.
   @param item An object that may have an attribute.
   @param attrib The attribute to be tested.
   @return true if the attribute has been given to the object.


*/
FALCON_FUNC  testAttribute( ::Falcon::VMachine *vm )
{
   Item *itm = vm->param( 0 );
   Item *attr = vm->param( 1 );

   if ( !itm->isObject() || ! attr->isAttribute() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "O,a" ) ) );
      return;
   }

   vm->regA().setBoolean( (itm->asObject()->has( attr->asAttribute() )) );
}

/*#
   @function giveTo
   @inset attrib_model
   @brief Gives a certain attribute to a certain object.
   @param attrib The attribute to be given
   @param obj The object that will receive the attribute

   This function is equivalent to the @b give statement, and is provided to allow
   functional processing of attributes. In example:

   @code
   attributes: opened
   dolist( [giveTo, opened], [obj1, obj2, obj3] )
   @endcode

   If the target object had already the attribute, nothing is done.
   If the first parameter is not an attribute or the second parameter is not an
   object, a ParamError is rasied.

   @see dolist
*/
FALCON_FUNC  giveTo( ::Falcon::VMachine *vm )
{
   Item *i_attrib = vm->param( 0 );
   Item *i_obj = vm->param( 1 );

   if ( i_attrib == 0 || ! i_attrib->isAttribute() ||
        i_obj == 0 || ! i_obj->isObject() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "a,X" ) ) );
      return;
   }

   vm->retval( (int64) (i_attrib->asAttribute()->giveTo( i_obj->asObject() ) ? 1 : 0) );
}


/*#
   @function removeFrom
   @inset attrib_model
   @brief Removes a certain attribute from a certain object.
   @param attrib The attribute to be removed
   @param obj The object from which the attribute must be removed

   This function is equivalent to the give statement using to remove
   an attribute, and is provided to allow functional processing of
   attributes. In example:

   @code
   attributes: opened
   dolist( [removeFrom, opened], [obj1, obj2, obj3] )
   @endcode

   If the target object didn't have the attribute, nothing is done.
   If the first parameter is not an attribute or the second parameter
   is not an object, a ParamError is rasied.
*/

FALCON_FUNC  removeFrom( ::Falcon::VMachine *vm )
{
   Item *i_attrib = vm->param( 0 );
   Item *i_obj = vm->param( 1 );

   if ( i_attrib == 0 || ! i_attrib->isAttribute() ||
        i_obj == 0 || ! i_obj->isObject() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "a,X" ) ) );
      return;
   }

   vm->retval( (int64) (i_attrib->asAttribute()->removeFrom( i_obj->asObject() ) ? 1 : 0) );
}

/*#
   @function removeFromAll
   @inset attrib_model
   @brief Removes a certain attribute from all the objects currently having it.
   @param attrib The attribute to be removed

   After this function is called, the target attribute will not be
   found in any object anymore.
*/

FALCON_FUNC  removeFromAll( ::Falcon::VMachine *vm )
{
   Item *i_attrib = vm->param( 0 );

   if ( i_attrib == 0 || ! i_attrib->isAttribute() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "a" ) ) );
      return;
   }

   i_attrib->asAttribute()->removeFromAll();
}


extern "C" static bool broadcast_next_attrib_next( ::Falcon::VMachine *vm )
{
   // break the chain if last call returned true or an OOB item
   if ( vm->regA().isOob() || vm->regA().isTrue() )
   {
      return false;
   }

   AttribObjectHandler *ho= (AttribObjectHandler *) vm->local(0)->asUserPointer();
   while ( ho != 0 )
   {
      CoreObject *obj = ho->object();
      // we want a copy anyhow...
      Item callable;
      obj->getProperty( vm->param(0)->asAttribute()->name(), callable );
      if ( callable.isCallable() )
      {
         // prepare our next frame
         vm->local(0)->setUserPointer( ho->next() );

         // great, we found an object willing to be called
         // prepare the stack;
         uint32 pc = vm->paramCount();
         for( uint32 i = 1; i < pc; i ++ )
         {
            vm->pushParameter( *vm->param( i ) );
         }
         callable.methodize( obj );
         vm->callFrame( callable, pc - 1 );
         return true;
      }
      ho = ho->next();
   }

   // we're done, return false
   return false;
}

FALCON_FUNC broadcast_next_attrib( ::Falcon::VMachine *vm )
{
   Attribute *attrib = vm->param(0)->asAttribute();

   // prevent making the frame (and calling) if empty
   if ( attrib->empty() )
      return;

   // signal we'll be using an attribute
   vm->addLocals( 1 );
   vm->local(0)->setUserPointer( attrib->head() );
   // fake a return true
   vm->retval( false );
   vm->returnHandler( broadcast_next_attrib_next );
}

extern "C" static bool broadcast_next_array( ::Falcon::VMachine *vm )
{
   // break chain if last call returned true
   if ( vm->regA().isOob() || vm->regA().isTrue() )
   {
      return false;
   }

   // select next item in the array.
   uint32 pos = (uint32) vm->local(0)->asInteger();
   CoreArray *aarr = vm->param(0)->asArray();

   // time to scan for a new attribute
   if ( pos >= aarr->length() )
   {
      // we're done
      return false;
   }

   // is it REALLY an attribute?
   const Item &attrib = aarr->at(pos);
   if ( ! attrib.isAttribute() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_type ).extra( "not an attribute" ) ) );
      return false;
   }

   // save next pos
   vm->local(0)->setInteger( pos + 1 );

   // select next item in the array
   vm->pushParameter( attrib );

   // and append our parameters
   uint32 pc = vm->paramCount();
   for( uint32 i = 1; i < pc; i++)
   {
      vm->pushParameter( *vm->param(i) );
   }

   // we pre-cached our frame initializer (broadcast_next_attrib)
   vm->callFrame( *vm->local(1), pc );
   return true;

}

/*#
   @function broadcast
   @inset attrib_model
   @param signaling An attribute or an array of attributes to be broadcast.
   @param ... Zero or more data to be broadcaset.
   @return Value returned by a message handler or nil.
   @brief Send a message to every object having an attribute.

   This function iterates over all the items having a certain attribute; if those objects provide a method
   named exactly as the attribute, then that method is called. A method can declare that it has “consumed”
   the message (i.e. done what is expected to be done) by returning true. In this case, the call chain is
   interrupted and broadcast returns. A method not wishing to prevent other methods to receive the incoming
   message must return false. Returning true means “yes, I have handled this message,
   no further processing is needed”.

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

   It is possible to receive more than one broadcast in the same handler using the “same handler idiom”:
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
   Item *i_attrib = vm->param( 0 );
   if ( ! i_attrib->isAttribute() && ! i_attrib->isArray() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "a|A,..." ) ) );
      return;
   }

   if ( i_attrib->isAttribute() )
   {
      Attribute *attrib = i_attrib->asAttribute();
      // nothing to broadcast?
      if ( attrib->empty() )
         return;

      // signal we'll be using an attribute
      vm->addLocals( 1 );
      vm->local(0)->setUserPointer( attrib->head() );
      vm->returnHandler( broadcast_next_attrib_next );
   }
   else
   {
      // prevent overdoing for nothing
      if ( i_attrib->asArray()->length() == 0 )
         return;

      // add space for the tracer
      vm->addLocals( 2 );
      vm->local(0)->setInteger( 0 );
      // pre-cache our service function
      Item *bcast_func = vm->findWKI( "%broadcast_next_attrib" );
      fassert( bcast_func != 0 );
      *vm->local(1) = *bcast_func;

      // set A to true to force first execution
      vm->returnHandler( broadcast_next_array );
   }

   // force vm to start first loop
   vm->retval( false );
}

}
}

/* end of attribute_ext.cpp */
