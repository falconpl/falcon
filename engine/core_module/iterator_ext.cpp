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

#include "core_module.h"
#include <falcon/sequence.h>
#include <falcon/falconobject.h>

namespace Falcon {
namespace core {

/*#
   @class Iterator
   @brief Indirect pointer to sequences.
   @ingroup general_purpose
   @param collection The collection on which to iterate.
   @optparam position Indicator for start position.

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
*/

/*#
   @init Iterator
   @brief Initialize the iterator

   The iterator is normally created at the begin of the sequence.
   If items in the collection can be directly accessed
      (i.e. if the collection is an array or a string), the @b position
      parameter can be any valid index.

   Otherwise, @b position can be 0 (the default) or -1. If it's -1,
   an iterator pointing to the last element of the collection will be returned.
*/

FALCON_FUNC  Iterator_init( ::Falcon::VMachine *vm )
{
   Item *collection = vm->param(0);
   Item *pos = vm->param(1);

   if( collection == 0 || ( pos != 0 && ! pos->isOrdinal() ) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X,[N]" ) ) );
      return;
   }

   FalconObject *self = dyncast<FalconObject *>( vm->self().asObject() );
   int32 p = pos == 0 ? 0: (int32) pos->forceInteger();

   switch( collection->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item tgt;
         String *orig = collection->asString();
         vm->referenceItem( tgt, *collection );
         self->setProperty( "_origin", tgt );

         if( orig->checkPosBound( p ) )
         {
            self->setProperty( "_pos", (int64) p );
         }
         else {
            vm->raiseRTError( new AccessError( ErrorParam( e_inv_params ) ) );
            return;
         }
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         Item tgt;
         MemBuf *orig = collection->asMemBuf();
         vm->referenceItem( tgt, *collection );
         self->setProperty( "_origin", tgt );
         int64 len = (int64) orig->length();
         if( p >= 0 && p < len )
         {
            self->setProperty( "_pos", (int64) p );
         }
         else if ( p < 0 && p >= -len )
         {
            self->setProperty( "_pos", (int64) len - p );
         }
         else {
            vm->raiseRTError( new AccessError( ErrorParam( e_inv_params ) ) );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *orig = collection->asArray();
         self->setProperty( "_origin", *collection );
         if( orig->checkPosBound( p ) )
         {
            self->setProperty( "_pos", (int64) p );
         }
         else {
            vm->raiseRTError( new AccessError( ErrorParam( e_inv_params ) ) );
            return;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *orig = collection->asDict();
         self->setProperty( "_origin", *collection );
         DictIterator *iter;
         if( p == 0 )
            iter = orig->first();
         else if( p == -1 )
            iter = orig->last();
         else {
            vm->raiseRTError( new AccessError( ErrorParam( e_inv_params ) ) );
            return;
         }

         self->setUserData( iter );
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         // Objects can have iterators if they have sequence extensions.
         CoreObject *obj = collection->asObject();

         Sequence *seq = obj->getSequence();
         if ( seq != 0 )
         {
            self->setProperty( "_origin", *collection );
            CoreIterator *iter = seq->getIterator( p != 0 );
            self->setUserData( iter );
            return;
         }
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ) ) );
      }
      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ) ) );
   }
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         int64 p = pos.forceInteger();
         vm->retval( p >= 0 && ( p < porigin->asString()->length() ) );
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         int64 p = pos.forceInteger();
         vm->retval( p >= 0 && ( p < porigin->asMemBuf()->length() ) );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         int64 p = pos.forceInteger();
         vm->retval( p >= 0 && ( p < porigin->asArray()->length() ) );
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         vm->retval( (int64) ( iter != 0 && iter->isValid() ? 1: 0 ) );
      }
   }
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         int64 p = pos.forceInteger();
         vm->regA().setBoolean( p >= 0 && (p + 1 < porigin->asString()->length() ) );
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         int64 p = pos.forceInteger();
         vm->regA().setBoolean( p >= 0 && (p + 1 < porigin->asMemBuf()->length() ) );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         int64 p = pos.forceInteger();
         vm->regA().setBoolean( p >= 0 && (p + 1 < porigin->asArray()->length() ) );
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         vm->regA().setBoolean( iter != 0 && iter->hasNext() );
      }
   }
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         vm->regA().setBoolean( pos.forceInteger() >= 0 );
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         vm->regA().setBoolean( iter != 0 && iter->hasPrev() );
      }
   }
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         if ( p < porigin->asString()->length() )
         {
            p++;
            vm->retval( p != porigin->asString()->length() );
            self->setProperty( "_pos", (int64) p );
         }
         else
            vm->regA().setBoolean( false );
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         if ( p < porigin->asMemBuf()->length() )
         {
            p++;
            vm->regA().setBoolean( p != porigin->asMemBuf()->length() );
            self->setProperty( "_pos", (int64) p );
         }
         else
            vm->regA().setBoolean( false );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         if ( p < porigin->asArray()->length() )
         {
            p++;
            vm->regA().setBoolean( p != porigin->asArray()->length() );
            self->setProperty( "_pos", (int64) p );
         }
         else
            vm->regA().setBoolean( false );
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         vm->regA().setBoolean( iter != 0 && iter->next() );
      }
   }
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         int64 p = pos.forceInteger();
         if ( p >= 0 )
         {
            self->setProperty( "_pos", p - 1 );
            vm->regA().setBoolean( p != 0 );
         }
         else
            vm->regA().setBoolean( false );
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         vm->regA().setBoolean( iter != 0 && iter->prev() );
      }
   }
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();
   Item *subst = vm->param( 0 );

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );

         if ( pos.forceInteger() < 0 )
            break;

         uint32 p = (uint32) pos.forceInteger();
         if( p < porigin->asString()->length() )
         {
            CoreString *str = new CoreString( 
               porigin->asString()->subString( p, p + 1 ) );
            vm->retval( str );

            // change value
            if( subst != 0 )
            {
               switch( subst->type() )
               {
                  case FLC_ITEM_STRING:
                     porigin->asString()->change( p, p + 1, *subst->asString() );
                  break;

                  case FLC_ITEM_NUM:
                     porigin->asString()->setCharAt( p, (uint32) subst->asNumeric() );
                  break;

                  case FLC_ITEM_INT:
                     porigin->asString()->setCharAt( p, (uint32) subst->asInteger() );
                  break;

                  default:
                     vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "S/N" ) ) );
               }
            }
            return;
         }
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         if ( pos.forceInteger() < 0 )
            break;
         uint32 p = (uint32) pos.forceInteger();
         if( p < porigin->asMemBuf()->length() )
         {
            vm->retval( (int64) porigin->asMemBuf()->get( p ) );
            // change value
            if( subst != 0 )
            {
               if ( subst->isOrdinal() )
                  porigin->asMemBuf()->set( p, (uint32) subst->forceInteger() );
               else
                  break; // only numbers allowed
            }
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         if ( pos.forceInteger() < 0 )
            break;
         uint32 p = (uint32) pos.forceInteger();
         if( p < porigin->asArray()->length() )
         {
            vm->retval( porigin->asArray()->at( p ) );
            // change value
            if( subst != 0 )
            {
               porigin->asArray()->at( p ) = *subst;
            }
            return;
         }
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         if( iter->isValid() )
         {
            vm->retval( iter->getCurrent() );
            // change value
            if( subst != 0 )
            {
               iter->getCurrent() = *subst;
            }

            return;
         }
      }
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_arracc ).extra( "Iterator.value" ) ) );
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   if( origin.isDict() )
   {
      DictIterator *iter = dyncast<DictIterator *>( self->getFalconData() );
      if( iter->isValid() )
      {
         vm->retval( iter->getCurrentKey() );
         return;
      }
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_arracc ).extra( "missing key" ) ) );
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

   if( i_other == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "O" ) ) );
      return;
   }

   if( i_other->isObject() )
   {
      CoreObject *other = i_other->asObject();

      if( other->derivedFrom( "Iterator" ) )
      {
         CoreObject *self = vm->self().asObject();

         Item origin, other_origin;
         self->getProperty( "_origin", origin );
         other->getProperty( "_origin", other_origin );
         if( *origin.dereference() == *other_origin.dereference() )
         {
            switch( origin.type() )
            {
               case FLC_ITEM_STRING:
               case FLC_ITEM_MEMBUF:
               case FLC_ITEM_REFERENCE:
               case FLC_ITEM_ARRAY:
               {
                  Item pos1, pos2;
                  self->getProperty( "_pos", pos1 );
                  other->getProperty( "_pos", pos2 );
                  vm->regA().setInteger( pos1.forceInteger() - pos2.forceInteger() );
               }
               return;

               default:
               {
                  CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
                  CoreIterator *other_iter = dyncast<CoreIterator *>( other->getFalconData() );
                  if( iter->equal( *other_iter ) )
                  {
                     vm->regA().setInteger( 0 );
                     return;
                  }
               }
               break;

            }
         }
      }
   }

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
   CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
   CoreIterator *iclone;

   // copy low level iterator, if we have one
   if ( iter != 0 ) {
      iclone = static_cast<CoreIterator *>(iter->clone());
      if ( iclone == 0 )
      {
         // uncloneable iterator
         vm->raiseError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }
   }
   else {
      iclone = 0;
   }

   // create an instance
   Item *i_cls = vm->findWKI( "Iterator" );
   fassert( i_cls != 0 );
   CoreObject *other = i_cls->asClass()->createInstance(iclone);

   // then copy properties
   Item prop;
   self->getProperty( "_origin", prop );
   other->setProperty( "_origin", prop );
   self->getProperty( "_pos", prop );
   other->setProperty( "_pos", prop );

   // we can return the object
   vm->retval( other );
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
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         String *str = porigin->asString();

         if ( p < str->length() )
         {
            str->remove( p, 1 );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         CoreArray *array = porigin->asArray();

         if ( p < array->length() )
         {
            array->remove( p );
            return;
         }
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         if( iter->isValid() )
         {
            iter->erase();
            return;
         }
      }
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_arracc ).extra( "Iterator.erase" ) ) );
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
   Item *i_key = vm->param(0);

   if( i_key == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   if ( porigin->isDict() )
   {
      DictIterator *iter = dyncast<DictIterator *>( self->getFalconData() );
      CoreDict* dict = porigin->asDict();

      if( iter->isOwner( dict ) )
      {
         vm->regA().setBoolean( dict->find( *i_key, *iter ) );
         return;
      }
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_arracc ).extra( "Iterator.find" ) ) );
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
   Item *i_key = vm->param(0);
   Item *i_value = vm->param(1);

   if( i_key == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   Item origin, *porigin;
   self->getProperty( "_origin", origin );
   porigin = origin.dereference();

   switch( porigin->type() )
   {
      case FLC_ITEM_STRING:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         String *str = porigin->asString();

         if ( ! i_key->isString() )
         {
            vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "S" ) ) );
            return;
         }

         if ( p < str->length() )
         {
            str->insert( p, 0, *i_key->asString() );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         Item pos;
         self->getProperty( "_pos", pos );
         uint32 p = (uint32) pos.forceInteger();
         CoreArray *array = porigin->asArray();

         if ( p < array->length() )
         {
            array->insert( *i_key, p );
            return;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         if ( i_value == 0 )
         {
            vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X,X" ) ) );
            return;
         }

         DictIterator *iter = dyncast<DictIterator *>( self->getFalconData() );
         CoreDict *dict = porigin->asDict();

         if( iter->isOwner( dict ) && iter->isValid() )
         {
            dict->smartInsert( *iter, *i_key, *i_value );
            return;
         }
      }
      break;

      default:
      {
         CoreIterator *iter = dyncast<CoreIterator *>( self->getFalconData() );
         if( iter->insert( *i_key ) )
         {
            return;
         }
      }
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_arracc ).extra( "Iterator.insert" ) ) );
}


/*#
   @method getOrigin Iterator
   @brief Returns the underlying sequence.
   @return The sequence being pointed by this iterator.
*/
FALCON_FUNC  Iterator_getOrigin( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->getProperty( "_origin", vm->regA() );
}

}
}

/* end of iterator_ext.cpp */
