/*
   FALCON - The Falcon Programming Language.
   FILE: iterator_ext.cpp

   Support for language level iterators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

#include "core_module.h"
#include <falcon/sequence.h>
#include <falcon/falconobject.h>
#include <falcon/iterator.h>
#include <falcon/itemid.h>

namespace Falcon {
namespace core {

/*#
   @class Iterator
   @brief Indirect pointer to sequences.
   @ingroup general_purpose
   @param collection The collection on which to iterate.
   @optparam atEnd Indicator for start position.

   An iterator is an object meant to point to a certain position in a collection
   (array, dictionary, string, or eventually user defined types), and to access
   iteratively the elements in that collection.

   Iterators may be used to alter a collection by removing the item they are pointing
   to, or by changing its value. They can be stored to be used at a later moment, or
   they can be passed as parameters. Most notably, they hide the nature of the underlying
   collection, so that they can be used as an abstraction layer to access underlying data,
   one item at a time.

   Altering the collection may cause an iterator to become invalid; only performing write
   operations through an iterator it is possible to guarantee that it will stay valid
   after the modify. A test for iterator validity is performed on each operation, and in
   case the iterator is not found valid anymore, an error is raised.

   Iterators supports equality tests and provide an equal() method. Two iterators pointing
   to the same element in the same collection are considered equal; so it is possible to
   iterate through all the items between a start and an end.

   @section init Initialization

   The iterator is normally created at the begin of the sequence.
   If items in the collection can be directly accessed

   If @b position is given and true, the iterator starts from the end
   of the sequence (that is, pointing to the last valid element in the
   sequence), otherwise it points at the first valid element.

*/

FALCON_FUNC  Iterator_init( ::Falcon::VMachine *vm )
{
   Item *collection = vm->param(0);

   if( collection == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin( e_orig_runtime )
         .extra( "Sequence,[B]" ) );
      return;
   }

   Iterator* iter;
   bool fromEnd = vm->param(1) != 0 && vm->param(1)->isTrue();

   switch( collection->type() )
   {
   case FLC_ITEM_ARRAY:
      iter = new Iterator( &collection->asArray()->items(), fromEnd );
      break;

   case FLC_ITEM_DICT:
      iter = new Iterator( &collection->asDict()->items(), fromEnd );
      break;

   case FLC_ITEM_OBJECT:
      {
         CoreObject* obj = collection->asObjectSafe();
         Sequence* seq = obj->getSequence();
         if( seq != 0 )
         {
            iter = new Iterator( seq, fromEnd );
         }
         else {
            throw new ParamError( ErrorParam( e_inv_params )
                     .origin( e_orig_runtime )
                     .extra( "Sequence,[B]" ) );
         }
      }
   break;

   default:
      throw new ParamError( ErrorParam( e_inv_params )
                           .origin( e_orig_runtime )
                           .extra( "Sequence,[B]" ) );
   }

   FalconObject *self = dyncast<FalconObject *>( vm->self().asObject() );
   self->setUserData( iter );
}


/*#
   @method hasCurrent Iterator
   @brief Check if the iterator is valid and can be used to
          access the underlying collection.
   @return true if the iterator is valid and can be used to
          access a current item.
*/

FALCON_FUNC  Iterator_hasCurrent( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );
   vm->regA().setBoolean( iter->hasCurrent() );
}

/*#
   @method hasNext Iterator
   @brief Check if the iterator is valid and a @a Iterator.next operation would
          still leave it valid.
   @return true if there is an item past to the current one.
*/

FALCON_FUNC  Iterator_hasNext( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );
   vm->regA().setBoolean( iter->hasNext() );
}

/*#
   @method hasPrev Iterator
   @brief Check if the iterator is valid and a @a Iterator.prev operation would
          still leave it valid.
   @return true if there is an item before to the current one.
*/
FALCON_FUNC  Iterator_hasPrev( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );
   vm->regA().setBoolean( iter->hasPrev() );
}

/*#
   @method next Iterator
   @brief Advance the iterator.
   @return true if the iterator is still valid after next() has been completed.

   Moves the iterator to the next item in the collection.
   If the iterator is not valid anymore, or if the current element was the last
   in the collection, the method returns false.
   If the iterator has successfully moved, it returns true.
*/
FALCON_FUNC  Iterator_next( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );
   if( ! iter->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );

   vm->regA().setBoolean( iter->next() );
}


/*#
   @method prev Iterator
   @brief Move the iterator back.
   @return true if the iterator is still valid after prev() has been completed.

   Moves the iterator to the previous item in the collection.
   If the iterator is not valid anymore, or if the current element was the
   first in the collection, the method returns false. If the iterator has
   successfully moved, it returns true.
*/
FALCON_FUNC  Iterator_prev( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );
   if( ! iter->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );

   vm->regA().setBoolean( iter->prev() );
}

/*#
   @method value Iterator
   @brief Retreives the current item in the collection.
   @optparam subst New value for the current item.
   @return The current item.
   @raise AccessError if the iterator is not valid.

   If the iterator is valid, the method returns the value of
   the item being currently pointed by the iterator.

   If a parameter @b subst is given, the current value in the sequence
   is changed to @b substs.
*/
FALCON_FUNC  Iterator_value( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );
   if( ! iter->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );

   Item* i_subst = vm->param(0);
   if( i_subst != 0 )
      iter->getCurrent() = *i_subst;
   else
      vm->regA() = iter->getCurrent();
}

/*#
   @method key Iterator
   @brief Retreives the current key in the collection.
   @return The current key.
   @raise AccessError if the iterator is not valid, or if the collection has not keys.

   If this iterator is valid and is pointing to a collection that provides key
   ordering (i.e. a dictionary),  it returns the current key; otherwise,
   it raises an AccessError.
*/

FALCON_FUNC  Iterator_key( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );
   if( ! iter->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );

   vm->regA() = iter->getCurrentKey();
}


/*#
   @method equal Iterator
   @param item The item to which this iterator must be compared.
   @brief Check if this iterator is equal to the provided item.
   @return True if the item matches this iterator.

   This method overrides the FBOM equal method, and overloads
   the equality check of the VM.
*/

FALCON_FUNC  Iterator_compare( ::Falcon::VMachine *vm )
{
   Item *i_other = vm->param(0);
   CoreObject *self = vm->self().asObject();
   Iterator* iter = dyncast<Iterator *>( self->getFalconData() );

   if( i_other == 0 || ! i_other->isOfClass( "Iterator" ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "Iterator" ).origin( e_orig_runtime ) );
   }

   Iterator* other = static_cast<Iterator*>( i_other->asObjectSafe()->getUserData() );

   if( ! iter->isValid() || ! other->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );

   if( other->equal(*iter) )
      vm->retval( (int64) 0 );   // declare them equal
   else
      // let the standard algorithm to decide.
      vm->retnil();
}

/*#
   @method clone Iterator
   @brief Returns an instance of this iterator pointing to the same item.
   @return A new copy of this iterator.

   Creates an iterator equivalent to this one. In this way, it is possible
   to record a previous position and use it later. Using a normal assignment
   wouldn't work, as the assignand would just be given the same iterator, and
   its value would change accordingly with the other image of the iterator.

*/

FALCON_FUNC  Iterator_clone( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator *iter = dyncast<Iterator *>( self->getFalconData() );
   if( ! iter->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );

   Item* i_itercls = vm->findWKI( "Iterator" );
   fassert( i_itercls != 0 && i_itercls->isClass() );
   CoreClass* Itercls = i_itercls->asClass();
   CoreObject* io = Itercls->createInstance();
   io->setUserData( iter->clone() );

   vm->retval( io );
}

/*#
   @method erase Iterator
   @brief Erase current item in the underlying sequence.
   @raise AccessError if the iterator is invalid.

   If the iterator is valid, this method removes current item. The iterator
   is moved to the very next item in the collection, and this may invalidate
   it if the removed element was the last one. To remove element while performing
   a scanning from the last element to the first one, remember to call the prev()
   method after every remove(); in forward scans, a successful remove() implies
   that the caller must not call next() to continue the scan.
*/
FALCON_FUNC  Iterator_erase( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator *iter = dyncast<Iterator *>( self->getFalconData() );
   if( ! iter->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );
   iter->erase();
}

/*#
   @method find Iterator
   @param key the key to be searched.
   @brief Moves this iterator on the searched item.
   @return true if the key was found, false otheriwse.
   @raise AccessError if the iterator is invalid or if the sequence doesn't provide keys.

   This method searches for an key in the underlying sequence, provided it offers search
   keys support. This is the case of the various dictionaries.

   This search is optimizied so that the subtree below the current position of the iterator
   is searched first. If the iterator is pointing to an item that matches the required
   key, this method returns immediately.

   After a succesful search, the iterator is moved to the position of the searched item.

   After a failed search, the iterator is moved to the smallest item in the sequence
   greater than the desired key; it's the best position for an insertion of the searched
   key.

   For example, to traverse all the items in a dictionary starting with 'C',
   the following code can be used:

   @code
   dict = [ "Alpha" => 1, "Beta" => 2, "Charlie" => 3, "Columbus" => 4, "Delta" => 5 ]
   iter = Iterator( dict )

   iter.find( "C" )  // we don't care if it succeeds
   while iter.hasCurrent() and iter.key()[0] == "C"
      > iter.key(), " => ", iter.value()
      iter.next()
   end
   @endcode

   Also, a failed search gives anyhow a useful hint position for a subsequent
   insertion, which may avoid performing the search again.
*/
FALCON_FUNC  Iterator_find( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator *iter = dyncast<Iterator *>( self->getFalconData() );
   Item* i_key = vm->param(0);

   if( i_key == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params )
                     .origin( e_orig_runtime )
                     .extra( "X" ) );
   }

   if( ! iter->isValid() )
      throw new ParamError( ErrorParam( e_invalid_iter )
               .origin( e_orig_runtime ) );

   if( iter->sequence()->isDictionary() )
   {
      ItemDict* dict = static_cast<ItemDict*>(iter->sequence());
      vm->regA().setBoolean( dict->findIterator( *i_key, *iter ) );
   }
   else
   {
      throw new ParamError( ErrorParam( e_non_dict_seq )
                     .origin( e_orig_runtime ) );
   }
}

/*#
   @method insert Iterator
   @param key Item to be inserted (or key, if the underlying sequence is keyed).
   @optparam value A value associated with the key.
   @brief Insert an item, or a pair of key values, in an underlying sequence.
   @raise AccessError if the iterator is invalid.

   Inserts an item at current position. In case the underlying sequence is an
   ordered sequence of key-value pairs, a correct position for insertion is first
   searched, and then the iterator is moved to the position of the inserted key.

   In this second case, if the iterator already points to a valid position for
   insertion of the given key, the search step is skipped.

   @see Iterator.find
*/

FALCON_FUNC  Iterator_insert( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Iterator *iter = dyncast<Iterator *>( self->getFalconData() );

   if( ! iter->isValid() )
        throw new ParamError( ErrorParam( e_invalid_iter )
                 .origin( e_orig_runtime ) );

   if( iter->sequence()->isDictionary() )
   {
      Item *i_key = vm->param(0);
      Item *i_value = vm->param(1);
      if( i_key == 0 || i_value == 0 )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "X,X" ) );
      }

      static_cast<ItemDict*>( iter->sequence() )->put( *i_key, *i_value );
   }
   else
   {
      Item *i_key = vm->param(0);
      if( vm->param(1) != 0 )
      {
         throw new ParamError( ErrorParam( e_non_dict_seq )
                              .origin( e_orig_runtime ) );
      }

      if( i_key == 0 )
      {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "X" ) );
      }


      iter->insert( *i_key );
   }
}

}
}

/* end of iterator_ext.cpp */
