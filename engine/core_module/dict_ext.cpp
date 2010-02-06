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
#include <falcon/coredict.h>
#include <falcon/iterator.h>
#include <falcon/vm.h>
#include <falcon/fassert.h>
#include <falcon/eng_messages.h>
#include <falcon/poopseq.h>
#include <falcon/garbagepointer.h>

namespace Falcon {
namespace core {

static void process_dictFrontBackParams( VMachine *vm, CoreDict* &dict, bool &bKey, bool &bRemove )
{
   if ( vm->self().isMethodic() )
   {
      dict = vm->self().asDict();
      bRemove = vm->param(0) != 0 && vm->param(0)->isTrue();
      bKey = vm->param(1) != 0 && vm->param(1)->isTrue();
   }
   else 
   {
      Item *i_dict = vm->param(0);
      if( i_dict == 0 || ! i_dict->isDict() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,[N,B,B]" ) );
      }
      
      dict = i_dict->asDict();
      bRemove = vm->param(1) != 0 && vm->param(1)->isTrue();
      bKey = vm->param(2) != 0 && vm->param(2)->isTrue();
   }
}

/****************************************
   Support for dictionaries
****************************************/

/*#
   @method front Dictionary
   @brief Returns the first item in the dictionary.
   @optparam remove If true, remove the dictionary entry too.
   @optparam key If true, return the key instead of the value.
   @return The first value (or key) in the dictionary.
   @raise AccessError if the dictionary is empty
*/

/*#
   @method back Dictionary
   @brief Returns the last item in the dictionary.
   @optparam remove If true, remove the dictionary entry too.
   @optparam key If true, return the key instead of the value.
   @return The last value (or key) in the dictionary.
   @raise AccessError if the dictionary is empty.
*/


/*#
   @funset core_dict_funcs Dictionary support
   @brief Dictionary related functions.
   @beginset core_dict_funcs
*/

/*#
   @function dictFront
   @brief Returns the first item in the dictionary.
   @param dict The dictionary on which to operate.
   @optparam remove If true, remove the dictionary entry too.
   @optparam key If true, return the key instead of the value.
   @return The first value (or key) in the dictionary.
   @raise AccessError if the dictionary is empty
*/
FALCON_FUNC  mth_dictFront( ::Falcon::VMachine *vm )
{
   CoreDict* dict; 
   bool bKey; 
   bool bRemove;
   
   process_dictFrontBackParams( vm, dict, bKey, bRemove );
   Iterator iter( &dict->items() );
   if ( bKey )
      vm->retval( iter.getCurrentKey() );
   else
      vm->retval( iter.getCurrent() );
   
   if ( bRemove )
      iter.erase();
}

/*#
   @function dictBack
   @brief Returns the last item in the dictionary.
   @param dict The dictionary on which to operate.
   @optparam remove If true, remove the dictionary entry too.
   @optparam key If true, return the key instead of the value.
   @return The last value (or key) in the dictionary.
   @raise AccessError if the dictionary is empty
*/
FALCON_FUNC  mth_dictBack( ::Falcon::VMachine *vm )
{
   CoreDict* dict; 
   bool bKey; 
   bool bRemove;
   
   process_dictFrontBackParams( vm, dict, bKey, bRemove );
   Iterator iter( &dict->items(), true );
   
   if ( bKey )
      vm->retval( iter.getCurrentKey() );
   else
      vm->retval( iter.getCurrent() );
   
   if ( bRemove )
      iter.erase();
}

/*#
   @method first Dictionary
   @brief Returns an iterator to the head of this dictionary.
   @return An iterator.
*/

FALCON_FUNC Dictionary_first( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   // we need to set the FalconData flag
   iterator->setUserData( new Iterator( &vm->self().asDict()->items() ) );
   vm->retval( iterator );
}

/*#
   @method last Dictionary
   @brief Returns an iterator to the head of this dictionary.
   @return An iterator.
*/

FALCON_FUNC Dictionary_last( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   // we need to set the FalconData flag
   iterator->setUserData( new Iterator( &vm->self().asDict()->items(), true  ) );
   vm->retval( iterator );
}
      
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( "D,[B]" ) );
   }

   bool mode = i_mode == 0 ? true: i_mode->isTrue();
   i_dict->asDict()->bless( mode );
   vm->regA() = *i_dict;
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

/*#
   @method remove Dictionary
   @brief Removes a given key from the dictionary.
   @param key The key to be removed
   @return True if the key is found and removed, false otherwise.

   If the given key is found, it is removed from the dictionary,
   and the function returns true. If it's not found, it returns false.
*/
FALCON_FUNC  mth_dictRemove ( ::Falcon::VMachine *vm )
{
   Item *dict, *key;
   
   if( vm->self().isMethodic() )
   {
      dict = &vm->self();
      key = vm->param(0);
   }
   else {
      dict = vm->param(0);
      key = vm->param(1);
   }
   
   if( dict == 0  || ! dict->isDict() || key == 0 ) 
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "X" : "D,X" ) );
   }

   CoreDict *d = dict->asDict();
   vm->regA().setBoolean( d->remove( *key ) );
}

/*#
   @function dictClear
   @brief Removes all the items from a dictionary.
   @param dict The dictionary to be cleared.
*/

/*#
   @method clear Dictionary
   @brief Removes all the items from this dictionary.
*/
FALCON_FUNC  mth_dictClear ( ::Falcon::VMachine *vm )
{
   Item *dict;
   
   if( vm->self().isMethodic() )
   {
      dict = &vm->self();
   }
   else {
      dict = vm->param(0);
      if( dict == 0  || ! dict->isDict() ) 
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "D"  ) );
      }
   }

   CoreDict *d = dict->asDict();
   d->clear();
}


/*#
   @function dictMerge
   @brief Merges two dictionaries.
   @param destDict The dictionary where the merge will take place.
   @param sourceDict A dictionary that will be inserted in destDict

   The function allows to merge two dictionaries.
*/

/*#
   @method merge Dictionary
   @brief Merges a dictionary into this one.
   @param sourceDict A dictionary that will be inserted in destDict
*/
FALCON_FUNC  mth_dictMerge ( ::Falcon::VMachine *vm )
{
   Item *dict1, *dict2;
   
   if( vm->self().isMethodic() )
   {
      dict1 = &vm->self();
      dict2 = vm->param(0);
   }
   else
   {
      dict1 = vm->param(0);
      dict2 = vm->param(1);
   }
   
   if( dict1 == 0 || ! dict1->isDict() 
      || dict2 == 0 || ! dict2->isDict() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( vm->self().isMethodic() ? "D" : "D,D" ) );
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

/*#
   @method keys Dictionary
   @brief Returns an array containing all the keys in this dictionary.
   @return An array containing all the keys.

   The returned keyArray contains all the keys in the dictionary. The values in
   the returned array are not necessarily sorted; however, they respect the
   internal dictionary ordering, which depends on a hashing criterion.

   If the dictionary is empty, then an empty array is returned.
*/
FALCON_FUNC  mth_dictKeys( ::Falcon::VMachine *vm )
{
   Item *i_dict;
   
   if( vm->self().isMethodic() )
   {
      i_dict = &vm->self();
   }
   else {
      i_dict = vm->param(0);
      if( i_dict == 0  || ! i_dict->isDict() ) 
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "D"  ) );
      }
   }

   CoreDict *dict = i_dict->asDict();
   CoreArray *array = new CoreArray;
   array->reserve( dict->length() );
   Iterator iter( &dict->items() );

   while( iter.hasCurrent() )
   {
      array->append( iter.getCurrentKey() );
      iter.next();
   }

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

/*#
   @method values Dictionary
   @brief Extracts all the values in this dictionary.
   @return An array containing all the values.

   The returned array contains all the value in the dictionary, in the same order by which
   they can be accessed traversing the dictionary.

   If the dictionary is empty, then an empty array is returned.
*/

FALCON_FUNC  mth_dictValues( ::Falcon::VMachine *vm )
{
   Item *i_dict;
   
   if( vm->self().isMethodic() )
   {
      i_dict = &vm->self();
   }
   else {
      i_dict = vm->param(0);
      if( i_dict == 0  || ! i_dict->isDict() ) 
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "D" ) );
      }
   }

   CoreDict *dict = i_dict->asDict();
   CoreArray *array = new CoreArray;
   array->reserve( dict->length() );
   Iterator iter( &dict->items() );

   while( iter.hasCurrent() )
   {
      array->append( iter.getCurrent() );
      iter.next();
   }

   vm->retval( array );
}

/*#
   @function dictFill
   @brief Fills the dictionary values with the given item.
   @param dict The array where to add the new item.
   @param item The item to be replicated.
   @return The same @b dict passed as parameter.

   This method allows to clear all the values in this dictionary, 
   resetting all the elements to a default value.
*/

/*#
   @method fill Dictionary
   @brief Fills the array with the given element.
   @param item The item to be replicated.
   @return This dictionary.

   This method allows to clear all the values in this dictionary, 
   resetting all the elements to a default value.
*/
FALCON_FUNC  mth_dictFill ( ::Falcon::VMachine *vm )
{
   Item *i_dict;
   Item *i_item;
   
   if ( vm->self().isMethodic() )
   {
      i_dict = &vm->self();
      i_item = vm->param(0);
   }
   else
   {
      i_dict = vm->param(0);
      i_item = vm->param(1);
   }

   if ( i_dict == 0 || ! i_dict->isDict() 
         || i_item == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "X" : "D,X" ) );
   }

   CoreDict *dict = i_dict->asDict();
   Iterator iter( &dict->items() );

   while( iter.hasCurrent() )
   {
      if ( i_item->isString() )
         iter.getCurrent() = new CoreString( *i_item->asString() );
      else
         iter.getCurrent() = *i_item;
      
      iter.next();
   }
   
   vm->retval( dict );
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
/*#
   @method get Dictionary
   @brief Retreives a value associated with the given key
   @param key The key to be found.
   @return The value associated with a key, or an out-of-band nil if not found.

   Return the value associated with the key, if present, or one of the
   values if more than one key matching the given one is present. If
   not present, the value returned will be nil. Notice that nil may be also
   returned if the value associated with a given key is exactly nil. In
   case the key cannot be found, the returned value will be marked as OOB.
   
   @note This method bypassess getIndex__ override in blessed (POOP) dictionaries.

   @see oob
*/
FALCON_FUNC  mth_dictGet( ::Falcon::VMachine *vm )
{
   Item *i_dict, *i_key;
   
   if( vm->self().isMethodic() )
   {
      i_dict = &vm->self();
      i_key = vm->param(0);
   }
   else {
      i_dict = vm->param(0);
      i_key = vm->param(1);
   }
   
   if( i_dict == 0  || ! i_dict->isDict() || i_key == 0 ) 
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "X" : "D,X" ) );
   }

   CoreDict *dict = i_dict->asDict();
   Item *value = dict->find( *i_key );
   if ( value == 0 )
   {
      vm->retnil();
      vm->regA().setOob();
   }
   else
      vm->retval( *value );
}

/*#
   @function dictSet
   @brief Stores a value in a dictionary
   @param dict A dictionary.
   @param key The key to be found.
   @param value The key to be set.
   @return True if the value was overwritten, false if it has been inserted anew.
   
   @note This method bypassess setIndex__ override in blessed (POOP) dictionaries.

   @see oob
*/

/*#
   @function set Dictionary
   @brief Stores a value in a dictionary
   @param key The key to be found.
   @param value The key to be set.
   @return True if the value was overwritten, false if it has been inserted anew.
   
   @note This method bypassess setIndex__ override in blessed (POOP) dictionaries.

   @see oob
*/
FALCON_FUNC  mth_dictSet( ::Falcon::VMachine *vm )
{
   Item *i_dict, *i_key, *i_value;
   
   if( vm->self().isMethodic() )
   {
      i_dict = &vm->self();
      i_key = vm->param(0);
      i_value = vm->param(1);
   }
   else {
      i_dict = vm->param(0);
      i_key = vm->param(1);
      i_value = vm->param(2);
   }
   
   if( i_dict == 0  || ! i_dict->isDict() || i_key == 0 || i_value == 0 ) 
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "X,X" : "D,X,X" ) );
   }

   CoreDict *dict = i_dict->asDict();
   Item *value = dict->find( *i_key );
   if ( value == 0 )
   {
      vm->regA().setBoolean( false );
      dict->put( *i_key, *i_value );
   }
   else {
      vm->regA().setBoolean( true );
      *value = *i_value;
   }
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

/*#
   @method find Dictionary
   @brief Returns an iterator set to a given key.
   @param key The key to be found.
   @return An iterator to the found item, or nil if not found.

   If the key is found in the dictionary, an iterator pointing to that key is
   returned. It is then possible to change the value of the found item, insert one
   item after or before the returned iterator or eventually delete the key. If the
   key is not found, the function returns nil.
*/

FALCON_FUNC  mth_dictFind( ::Falcon::VMachine *vm )
{
   Item *i_dict, *i_key;
   
   if( vm->self().isMethodic() )
   {
      i_dict = &vm->self();
      i_key = vm->param(0);
   }
   else {
      i_dict = vm->param(0);
      i_key = vm->param(1);
   }
   
   if( i_dict == 0  || ! i_dict->isDict() || i_key == 0 ) 
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "X" : "D,X" ) );
   }

   CoreDict *dict = i_dict->asDict();
   Iterator iter( &dict->items() );

   if ( ! dict->findIterator( *i_key, iter ) )
      vm->retnil();
   else 
   {
      // find the iterator class, we'll need it
      Item *i_iclass = vm->findWKI( "Iterator" );
      fassert( i_iclass != 0 );
      
      CoreObject *ival = i_iclass->asClass()->createInstance( new Iterator( iter ) );
      ival->setProperty( "_origin", *i_dict );
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
   @endcode

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

/*#
   @method best Dictionary
   @brief Returns an iterator set to a given key, or finds the best position for its insertion.
   @param key The key to be found.
   @return An iterator to the best possible position.
   
   @see dictBest
*/
   
FALCON_FUNC  mth_dictBest( ::Falcon::VMachine *vm )
{
   Item *i_dict, *i_key;
   
   if( vm->self().isMethodic() )
   {
      i_dict = &vm->self();
      i_key = vm->param(0);
   }
   else {
      i_dict = vm->param(0);
      i_key = vm->param(1);
   }
   
   if( i_dict == 0  || ! i_dict->isDict() || i_key == 0 ) 
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( vm->self().isMethodic() ? "X" : "D,X" ) );
   }

   // find the iterator class, we'll need it
   Item *i_iclass = vm->findWKI( "Iterator" );
   fassert( i_iclass != 0 );

   CoreDict *dict = i_dict->asDict();
   Iterator* iter = new Iterator( &dict->items() );
   CoreObject *ival = i_iclass->asClass()->createInstance( iter );
   ival->setProperty( "_origin", *i_dict );

   vm->regA() = ival;
   if ( ! dict->findIterator( *i_key, *iter ) )
   {
      vm->regA().setOob();
   }
}

/*#
   @method comp Dictionary
   @brief Appends elements to this dictionary through a filter.
   @param source A sequence, a range or a callable generating items.
   @optparam filter A filtering function receiving one item at a time.
   @return This dictionary.

   Please, see the description of @a Sequence.comp.

   When the target sequence (this item) is a dictionary, each element that is to be
   appended must be exactly an array with two items; the first will be used as a key,
   and the second as the relative value.

   For example:

   @code
   dict = [=>].comp(
      // the source
      .[ 'bananas' 'skip me' 'apples' 'oranges' '<end>' 'melons' ],
      // the filter
      { element, dict =>
        if " " in element: return oob(1)
        if "<end>" == element: return oob(0)
        return [ "A" / len(dict), element ]   // (1)
      }
   )
   @endcode

   The element generated by the filter is a 2 element array, which is then
   stored in the dictionary as a pair of key-value items.

   @note In the example, the expression marked with (1) "A"/len(dict)
   causes the current number of elements in the dictionary to be added to the UNICODE
   value of the "A" letter; so the generated key will be "A" when the dictionary
   has zero elements, "B" when it has one element and so on.

   If the dictionary is blessed, then it is treated as an object, and instead of
   adding directly the pair of key/value items, it's @b append method is repeatedly
   called with the generated item as its parameter.
   In this case, the type and length of each element is not relevant.

   @see Sequence.comp
*/

FALCON_FUNC  Dictionary_comp ( ::Falcon::VMachine *vm )
{
   if ( vm->param(0) == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "R|A|C|Sequence, [C]" ) );
   }

   // Save the parameters as the stack may change greatly.
   CoreDict* dict = vm->self().asDict();

   Item i_gen = *vm->param(0);
   Item i_check = vm->param(1) == 0 ? Item(): *vm->param(1);

   // if this is a blessed dictionary, we must use the append method.
   if ( dict->isBlessed() )
   {
      // this will throw if dict has not "append"
      PoopSeq *seq = new PoopSeq( vm, dict );
      vm->pushParam( new GarbagePointer( seq ) );
      seq->comprehension_start( vm, dict, i_check );
   }
   else {
      dict->items().comprehension_start( vm, dict, i_check );
   }

   vm->pushParam( i_gen );
}

/*#
   @method mcomp Dictionary
   @brief Appends elements to this dictionary through a filter.
   @param ... One or more sequences, ranges or callables generating items.
   @optparam filter A filtering function receiving one item at a time.
   @return This dictionary.

   Please, see the description of @a Sequence.comp, and the general @a Dictionary.comp
   for dictioanry-specific notes.

   @see Sequence.mcomp
*/

FALCON_FUNC  Dictionary_mcomp ( ::Falcon::VMachine *vm )
{
   // Save the parameters as the stack may change greatly.
   CoreDict* dict = vm->self().asDict();
   StackFrame* current = vm->currentFrame();

   // if this is a blessed dictionary, we must use the append method.
   if ( dict->isBlessed() )
   {
      // this will throw if dict has not "append"
      PoopSeq *seq = new PoopSeq( vm, dict );
      vm->pushParam( new GarbagePointer( seq ) );
      seq->comprehension_start( vm, dict, Item() );
   }
   else {
      dict->items().comprehension_start( vm, dict, Item() );
   }

   for( uint32 i = 0; i < current->m_param_count; ++i )
   {
      vm->pushParam( current->m_params[i] );
   }
}

/*#
   @method mfcomp Dictionary
   @brief Appends elements to this dictionary through a filter.
   @param filter A filter function receiving each element before its insertion, or nil.
   @param ... One or more sequences, ranges or callables generating items.
   @optparam filter A filtering function receiving one item at a time.
   @return This dictionary.

   Please, see the description of @a Sequence.comp, and the general @a Dictionary.comp
   for dictioanry-specific notes.

   @see Sequence.mfcomp
*/

FALCON_FUNC  Dictionary_mfcomp ( ::Falcon::VMachine *vm )
{
   if ( vm->param(0) == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "C, ..." ) );
   }

   // Save the parameters as the stack may change greatly.
   CoreDict* dict = vm->self().asDict();
   StackFrame* current = vm->currentFrame();

   Item i_check = vm->param(0) == 0 ? Item(): *vm->param(10);

   // if this is a blessed dictionary, we must use the append method.
   if ( dict->isBlessed() )
   {
      // this will throw if dict has not "append"
      PoopSeq *seq = new PoopSeq( vm, dict );
      vm->pushParam( new GarbagePointer( seq ) );
      seq->comprehension_start( vm, dict, i_check );
   }
   else {
      dict->items().comprehension_start( vm, dict, i_check );
   }

   for( uint32 i = 1; i < current->m_param_count; ++i )
   {
      vm->pushParam( current->m_params[i] );
   }
}



}
}

/* end of dict.cpp */
