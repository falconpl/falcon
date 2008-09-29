/*
   FALCON - The Falcon Programming Language.
   FILE: dict.cpp

   Dictionary api
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio mar 16 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Dictionary api
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/vm.h>
#include <falcon/fassert.h>
#include "core_messages.h"

/*#

*/

/*#
   @funset core_dict_funcs Dictionary support
   @brief Dictionary related functions.
   @beginset core_dict_funcs
*/

namespace Falcon {
namespace core {

/****************************************
   Support for dictionaries
****************************************/

/*#
   @function bless
   @brief Blesses a dictionary, making it an OOP instance.
   @param dict A dictionary to be blessed.
   @optparam mode True (default) to bless the dictionary, false to unbless it.
   @return The same dictonary passed as @b dict.

   Blessed dictionaries become sensible to OOP operators: dot accessors
   and "provides" keyword behave as if the dictionary was an object instance,
   with its string entries being properties.
*/
FALCON_FUNC  bless ( ::Falcon::VMachine *vm )
{
   Item *i_dict = vm->param(0);
   Item *i_mode = vm->param(1);


   if( i_dict == 0  || ! i_dict->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
      origin( e_orig_runtime ).
      extra( "D,[B]" ) ) );
      return;
   }

   bool mode = i_mode == 0 ? true: i_mode->isTrue();
   i_dict->asDict()->bless( mode );
   vm->regA() = *i_dict;
}

/*#
   @function dictInsert
   @brief Inserts a new key/value pair in the dictionary
   @param dict a dictionary
   @param key the new key
   @param value the new value

   This functions adds a couple of key/values in the dictionary,
   the key being forcibly unique.

   If the given key is already present is found on the dictionary,
   the previous value is overwritten.

*/

FALCON_FUNC  dictInsert ( ::Falcon::VMachine *vm )
{
   Item *dict = vm->param(0);
   Item *key = vm->param(1);
   Item *value = vm->param(2);


   if( dict == 0  || ! dict->isDict() || key == 0 || value == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   dict->asDict()->insert( *key, *value );
}

/*#
   @function dictRemove
   @brief Removes a given key from the dictionary.
   @param dict A dictionary.
   @param key The key to be removed
   @return True if the key is found and removed, false otherwise.

   If the given key is found, it is removed from the dictionary,
   and the function returns true. If it's not found, it returns false.
*/
FALCON_FUNC  dictRemove ( ::Falcon::VMachine *vm )
{
   Item *dict = vm->param(0);
   Item *key = vm->param(1);

   if( dict == 0  || ! dict->isDict() || key == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *d = dict->asDict();
   vm->regA().setBoolean( d->remove( *key ) );
}

/*#
   @function dictClear
   @brief Removes all the items from a dictionary.
   @param dict The dictionary to be cleared.
*/

FALCON_FUNC  dictClear ( ::Falcon::VMachine *vm )
{
   Item *dict = vm->param(0);

   if( dict == 0  || ! dict->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *d = dict->asDict();
   d->clear();
}


/*#
   @function dictMerge
   @brief Merges two dictionaries.
   @param destDict The dictionary where the merge will take place.
   @param sourceDict A dictionary that will be inserted in destDict

   The function allows to merge two dictionaries. If the optional parameter is
   not given, or false, in case of keys in sourceDict matching the ones in
   destDict, the relative values will be overwritten with the ones coming from
   sourceDict.
*/
FALCON_FUNC  dictMerge ( ::Falcon::VMachine *vm )
{
   Item *dict1 = vm->param(0);
   Item *dict2 = vm->param(1);
   if( dict1 == 0 || ! dict1->isDict() || dict2 == 0 || ! dict2->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *d1 = dict1->asDict();
   CoreDict *d2 = dict2->asDict();
   d1->merge( *d2 );
}

/*#
   @function dictKeys
   @brief Returns an array containing all the keys in the dictionary.
   @param dict A dictionary.
   @return An array containing all the keys.

   The returned keyArray contains all the keys in the dictionary. The values in
   the returned array are not necessarily sorted; however, they respect the
   internal dictionary ordering, which depends on a hashing criterion.

   If the dictionary is empty, then an empty array is returned.
*/
FALCON_FUNC  dictKeys( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);

   if( dict_itm == 0 || ! dict_itm->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();
   CoreArray *array = new CoreArray( vm );
   array->reserve( dict->length() );
   DictIterator *iter = dict->first();

   while( iter->isValid() )
   {
      array->append( iter->getCurrentKey() );
      iter->next();
   }
   delete iter;

   vm->retval( array );
}

/*#
   @function dictValues
   @brief Extracts all the values in the dictionary.
   @param dict A dictionary.
   @return An array containing all the values.

   The returned array contains all the value in the dictionary, in the same order by which
   they can be accessed traversing the dictionary.

   If the dictionary is empty, then an empty array is returned.
*/

FALCON_FUNC  dictValues( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);

   if( dict_itm == 0 || ! dict_itm->isDict() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();
   CoreArray *array = new CoreArray( vm );
   array->reserve( dict->length() );
   CoreIterator *iter = dict->first();

   while( iter->isValid() )
   {
      array->append( iter->getCurrent() );
      iter->next();
   }
   delete iter;

   vm->retval( array );
}

/*#
   @function dictGet
   @brief Retreives a value associated with the given key
   @param dict A dictionary.
   @param key The key to be found.
   @return The value associated with a key, or an out-of-band nil if not found.

   Return the value associated with the key, if present, or one of the
   values if more than one key matching the given one is present. If
   not present, the value returned will be nil. Notice that nil may be also
   returned if the value associated with a given key is exactly nil. In
   case the key cannot be found, the returned value will be marked as OOB.

   @see oob
*/
FALCON_FUNC  dictGet( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);
   Item *key_item = vm->param(1);

   if( dict_itm == 0 || ! dict_itm->isDict() || key_item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   CoreDict *dict = dict_itm->asDict();
   Item *value = dict->find( *key_item );
   if ( value == 0 )
   {
      vm->retnil();
      vm->regA().setOob();
   }
   else
      vm->retval( *value );
}

/*#
   @function dictFind
   @brief Returns an iterator set to a given key.
   @param dict The dictionary.
   @param key The key to be found.
   @return An iterator to the found item, or nil if not found.

   If the key is found in the dictionary, an iterator pointing to that key is
   returned. It is then possible to change the value of the found item, insert one
   item after or before the returned iterator or eventually delete the key. If the
   key is not found, the function returns nil.
*/
FALCON_FUNC  dictFind( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);
   Item *key_item = vm->param(1);

   if( dict_itm == 0 || ! dict_itm->isDict() || key_item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // find the iterator class, we'll need it
   Item *i_iclass = vm->findWKI( "Iterator" );
   fassert( i_iclass != 0 );

   CoreDict *dict = dict_itm->asDict();

   DictIterator *value = dict->findIterator( *key_item );
   if ( value == 0 )
      vm->retnil();
   else {
      CoreObject *ival = i_iclass->asClass()->createInstance();
      ival->setProperty( "_origin", *dict_itm );
      ival->setUserData( value );
      vm->retval( ival );
   }
}

/*#
   @function dictBest
   @brief Returns an iterator set to a given key, or finds the best position for its insertion.
   @param dict The dictionary.
   @param key The key to be found.
   @return An iterator to the best possible position.

   If the key is found in the dictionary, an iterator pointing to that key is
   returned. It is then possible to change the value of the found item, insert one
   item after or before the returned iterator or eventually delete the key. If the
   key is not found, an iterator pointing to the first key greater than the
   searched one is returned. The position is so that an insertion there would place
   the key in the right order. If the key is not found, the returned iterator is
   marked as out-of-band (see oob() at page 14).

   The method insert() of the Iterator class is optimized so that if the
   iterator is already in a valid position where to insert its key, the binary
   search is not performed again. Compare:

   @code
   d = [ "a" => 1, "c"=>2 ]

   // two searches
   if "b" notin d
      d["b"] = 0
   else
      d["b"]++
   end

   // one search
   iter = dictBest( dict, "b" )
   isoob(iter) ? iter.insert( "b", 0 ) : iter.value( iter.value() + 1 )
   @code

   In the first case, the insertion of a special value in a dictionary where the
   value is still not present has required a first search then a second one at
   insertion or modify. In the second case, the iterator can use the position
   information it has stored to avoid a second search.

   This function can also be used just to know what is the nearest key being
   present in the dictionary. The searched key is greater than the one that can be
   reached with Iterator.prev(), and less or equal than the one pointed. If
   Iterator.hasPrev() is false, then the searched key is smaller than any other in
   the collection, and if Iterator.hasCurrent() is false, then the key is greater
   than any other.
*/

FALCON_FUNC  dictBest( ::Falcon::VMachine *vm )
{
   Item *dict_itm = vm->param(0);
   Item *key_item = vm->param(1);

   if( dict_itm == 0 || ! dict_itm->isDict() || key_item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // find the iterator class, we'll need it
   Item *i_iclass = vm->findWKI( "Iterator" );
   fassert( i_iclass != 0 );

   CoreDict *dict = dict_itm->asDict();
   DictIterator *value = dict->first();
   CoreObject *ival = i_iclass->asClass()->createInstance();
   ival->setProperty( "_origin", *dict_itm );
   ival->setUserData( value );
   vm->regA() = ival;
   if ( ! dict->find( *key_item, *value ) )
   {
      vm->regA().setOob();
   }
}

}
}

/* end of dict.cpp */
