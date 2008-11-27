/*
   FALCON - The Falcon Programming Language.
   FILE: array.cpp

   Array specialized operation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @funset core_array_funcs Array support
   @brief Array related functions.
   @beginset core_array_funcs
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/carray.h>
#include <falcon/vm.h>
#include "core_messages.h"

namespace Falcon {
namespace core {

/*#
   @function arrayIns
   @brief Inserts an item into an array.
   @param array The array where the item should be placed.
   @param itempos  The position where the item should be placed.
   @param item The item to be inserted.

   The item is inserted before the given position. If pos is 0, the item is
   inserted in the very first position, while if it's equal to the array length, it
   is appended at the array tail.
*/

FALCON_FUNC  arrayIns ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item_pos = vm->param(1);
   Item *item = vm->param(2);

   if ( array_x == 0 || array_x->type() != FLC_ITEM_ARRAY ||
         item_pos == 0 || ! item_pos->isOrdinal() ||
         item == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("A,N,X") ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   int32 pos = (int32) item_pos->forceInteger();

   if ( ! array->insert( *item, pos ) ) {
      // array access error
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) )
         );
   }
}

/*#
   @function arrayDel
   @brief Deletes the first element matching a given item.
   @param array The array that is to be changed.
   @param item The item that must be deleted.
   @return true if at least one item has been removed, false otherwise.

   The function scans the array searching for an item that is considered
   equal to the given one. If such an item can be found, it is removed and
   the function returns true. If the item cannot be found, false is returned.

   Only the first item matching the given one will be deleted.
*/
FALCON_FUNC  arrayDel ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item_rem = vm->param(1);

   if ( array_x == 0 || array_x->type() != FLC_ITEM_ARRAY ||
         item_rem == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   // find the element
   CoreArray *array = array_x->asArray();
   Item *elements = array->elements();
   for( uint32 i = 0; i < array->length(); i++ ) {
      if ( elements[i] == *item_rem ) {
         array->remove( i );
         vm->regA().setBoolean(true);
         return;
      }
   }

   vm->regA().setBoolean(false);
}

/*#
   @function arrayDelAll
   @brief Deletes all the occurrences of a given item in an array.
   @param array The array that is to be changed.
   @param item The item that must be deleted.
   @return true if at least one item has been removed, false otherwise.

   This function removes all the elements of the given array that are
   considered equivalent to the given item. If one or more elements have
   been found and deleted, the function will return true, else it will
   return false.
*/
FALCON_FUNC  arrayDelAll ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item_rem = vm->param(1);

   if ( array_x == 0 || array_x->type() != FLC_ITEM_ARRAY ||
         item_rem == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   // find the element
   bool done = false;
   CoreArray *array = array_x->asArray();
   Item *elements = array->elements();
   uint32 i = 0;
   while( i < array->length() )
   {
      if ( elements[i] == *item_rem ) {
         array->remove( i );
         done = true;
      }
      else
         i++;
   }

   vm->regA().setBoolean( done );
}

/*#
   @function arrayAdd
   @brief Adds an element to an array.
   @param array The array where to add the new item.
   @param element The item to be added.

   The element will be added at the end of the array,
   and its size will be increased by one.
*/
FALCON_FUNC  arrayAdd ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item = vm->param(1);

   if ( array_x == 0 || array_x->type() != FLC_ITEM_ARRAY )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   array->append( *item );
}

/*#
   @function arrayResize
   @brief Changes the size of the array.
   @param array The array that will be resize.
   @param newSize The new size for the array.

   If the given size is smaller than the current size, the array is
   shortened. If it's larger, the array is grown up to the desired size,
   and the missing elements are filled by adding nil values.
*/

FALCON_FUNC  arrayResize ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item_size = vm->param(1);

   if ( array_x == 0 || array_x->type() != FLC_ITEM_ARRAY ||
         item_size == 0 || !item_size->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   array->resize( (int32) item_size->forceInteger() );
}

/*#
   @function arrayRemove
   @brief Removes one or more elements in the array.
   @param array The array from which items must be removed.
   @param itemPos The position of the item to be removed, or the first of the items to be removed.
   @optparam lastItemPos The last item to be removed, in range semantic.

   Remove one item or a range of items. The size of the array is shortened accordingly.
   The semantic of @b lastItemPos is the same as ranged access to the array.
*/

FALCON_FUNC  arrayRemove( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item_start = vm->param(1);
   Item *item_end = vm->param(2);

   if ( array_x == 0 || !array_x->isArray() ||
         item_start == 0 || !item_start->isOrdinal() ||
         (item_end != 0 && !item_end->isOrdinal() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   int32 pos = (int32) item_start->forceInteger();

   bool res;
   if ( item_end != 0 ) {
      int32 posLast = (int32) item_end->forceInteger();
      res =  array->remove( pos, posLast );
   }
   else
      res = array->remove( pos );

   if ( ! res ) {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
   }
}

/*#
   @function arrayCopy
   @brief Adds an element to an array.
   @param array The original array.
   @optparam start If given, the first element from where to copy.
   @optparam end If given, one-past-last element to be copied.
   @return A shallow copy of the array or of part of it.

   Actually, this function calls the range-access operator with a complete
   interval ([:]) if neither start or end are given, an open interval ([start:]) if
   only start is given and a full range if both start and end are given. This means
   that the range operation rules apply 1:1 to this function. If end is smaller
   than start, the copy order is reversed, and if they are the same, an empty array
   is created. The end parameter, if given, must be one past the last element to be
   copied; so:

   @code
   ret = arrayCopy( source, 5, 6 )
   @endcode

   will create an array containing only the element at position 5 in the source
   array.
*/

FALCON_FUNC  arrayCopy( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item_start = vm->param(1);
   Item *item_end = vm->param(2);

   if ( array_x == 0 || !array_x->isArray() ||
         (item_start != 0 && !item_start->isOrdinal()) ||
         (item_end != 0 && ! (item_end->isOrdinal()||item_end->isNil()) ))
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_array_missing ) ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   int64 start = item_start == 0 ? 0 : item_start->forceInteger();
   int64 end = item_end == 0 || item_end->isNil() ? array->length() : item_end->forceInteger();
   CoreArray *arr1 = array->partition( (int32) start, (int32) end );
   if ( arr1 == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_array_missing ) ) ) );
      return;
   }
   vm->retval( arr1 );
}

/*#
   @function arrayBuffer
   @brief Creates an array filled with nil items.
   @param size The length of the returned array.
   @optparam defItem The default item to be that will fill the array.
   @return An array filled with @b size nil elements.

   This function is useful when the caller knows the number of needed items. In
   this way, it is just necessary to set the various elements to their values,
   rather than adding them to the array. This will result in faster operations.

   If @b defItem is not given, the elements in the returned array will be set to
   nil, otherwise each element will be a flat copy of @b item.
*/
FALCON_FUNC  arrayBuffer ( ::Falcon::VMachine *vm )
{
   Item *item_size = vm->param(0);
   Item *i_item = vm->param(1);

   if ( item_size == 0 || ! item_size->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N,[X]" ) ) );
      return;
   }

   int32 nsize = (int32) item_size->forceInteger();
   CoreArray *array = new CoreArray( vm, nsize );
   Item *mem = array->elements();

   if( i_item == 0 )
   {
      for ( int i = 0; i < nsize; i++ ) {
         mem[i].setNil();
      }
   }
   else 
   {
      for ( int i = 0; i < nsize; i++ ) {
         mem[i] = *i_item;
      }      
   }

   array->length( nsize );
   vm->retval( array );
}

/*#
   @function arrayHead
   @brief Extracts the first element of an array and returns it.
   @param array The array that will be modified.
   @return The extracted item.

   This function removes the first item of the array and returns it.
   If the original array is empty, AccessError is raised.
*/
FALCON_FUNC  arrayHead ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "A" ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   if ( array->length() == 0 )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_emptyarr ) ) ) );
      return;
   }

   vm->retval( array->at(0) );
   array->remove(0);
}

/*#
   @function arrayTail
   @brief Extracts the last element of an array and returns it.
   @param array The array that will be modified.
   @return The extracted item.

   This function removes the last item of the array and returns it.
   If the original array is empty, AccessError is raised.
*/


FALCON_FUNC  arrayTail ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "A" ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   if ( array->length() == 0 )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_emptyarr ) ) ) );
      return;
   }

   vm->retval( array->at(array->length()-1) );
   array->remove(array->length()-1);
}

/*#
   @function arrayFind
   @brief Searches for a given item in an array.
   @param array The array that will be searched.
   @param item The array that will be searched.
   @optparam start Optional first element to be searched.
   @optparam end Optional last element +1 to be searched.
   @return The position of the searched item in the array, or -1 if not found.

   This function searches the array for a given item (or for an item considered
   equivalent to the given one). If that item is found, the item position is
   returned, else -1 is returned.

   An optional range may be specified to limit the search in the interval start
   included, end excluded.
*/
FALCON_FUNC  arrayFind ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() || item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar1 ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar3 ) ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   // can find nothing in an empty array
   if ( array->length() == 0 ) {
      vm->retval( -1 );
      return;
   }

   int32 pos_start = (int32) (start == 0 ? 0 : start->asInteger());
   int32 pos_end = (int32) (end == 0 ? array->length() : end->asInteger());
   if ( pos_start > pos_end ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_scan_end ) ) ) );
      return;
   }

   if ( pos_start < 0 || pos_start >= (int32) array->length() || pos_end > (int32) array->length()) {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   Item *elements = array->elements();
   for( int32 i = pos_start; i < pos_end; i++ )
   {
      if ( vm->compareItems( elements[i], *item ) == 0 )
      {
         vm->retval(i);
         return;
      }
   }

   vm->retval(-1);
}

/*#
   @function arrayScan
   @brief Searches an array for an item satisfying arbitrary criteria.
   @param array The array that will be searched.
   @param scanFunc Function that verifies the criterion.
   @optparam start Optional first element to be searched.
   @optparam end Optional upper end of the range..
   @return The position of the searched item in the array, or -1 if not found.

   With this function, it is possible to specify an arbitrary search criterion.
   The items in the array are fed one after another as the single parameter to the
   provided scanFunc, which may be any Falcon callable item, including a method. If
   the scanFunc returns true, the scan is interrupted and the index of the item is
   returned. If the search is unsuccesful, -1 is returned.

   An optional start parameter may be provided to begin searching from a certain
   point on. If also an end parameter is given, the search is taken between start
   included and end excluded (that is, the search terminates when at the element
   before the position indicated by end).

   Scan function is called in atomic mode. The called function cannot be
   interrupted by external kind requests, and it cannot sleep or yield the
   execution to other coroutines.
*/
FALCON_FUNC  arrayScan ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *func_x = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_array_first ) ) ) );
      return;
   }

   if ( func_x == 0 || ! func_x->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_second_call ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar3 ) ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   // can find nothing in an empty array
   if ( array->length() == 0 ) {
      vm->retval( -1 );
      return;
   }

   int32 pos_start = (int32) (start == 0 ? 0 : start->asInteger());
   int32 pos_end = (int32) (end == 0 ? array->length() : end->asInteger());
   if ( pos_start > pos_end ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_scan_end ) ) ) );
      return;
   }

   if ( pos_start < 0 || pos_start >= (int32) array->length() ||
         pos_end > (int32) array->length()) {
       vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   Item *elements = array->elements();
   // fetching as we're going to change the stack
   Item func = *func_x;
   for( int32 i = pos_start; i < pos_end; i++ )
   {
      vm->pushParameter( elements[i] );
      vm->callItemAtomic( func, 1 );
      if ( vm->regA().isTrue() ) {
         vm->retval( i );
         return;
      }
   }

   vm->retval(-1);
}


/*#
   @function arrayFilter
   @brief Filters an array using a given function.
   @param array The array to be filtered.
   @param filterFunc A function to filter the array
   @optparam start Optional first element to be filtered.
   @optparam end Optional last element +1 to be filtered.
   @return an array with some of the items removed.

   This function perform what is usually defined a “filtering” of the source structure
   into a target array. The filterFunc parameter is a user-provide callable item
   that will be called once for each item in the specified array range;
   if the function returns a true value, the item is copied in the returned array,
   otherwise it is ignored. The filter function may actually be any kind of
   callable, including methods, Sigmas and external functions.

   In case the function never returns true, an empty array is returned.

   A start-end range may optionally be specified to limit the filtering to a
   part of the array. As for other functions in this module, the range will
   cause filtering for the items from start to end-1.

   The filter function is called in atomic mode. The called function cannot be
   interrupted by external kind requests, and it cannot sleep or yield the
   execution to other coroutines.

   @note This function is superseeded by @a filter and may be removed
   in future versions.
*/

FALCON_FUNC  arrayFilter( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *func_x = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_array_first ) ) ) );
      return;
   }

   if ( func_x == 0 || ! func_x->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_second_call ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar3 ) ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   // can find nothing in an empty array
   if ( array->length() == 0 ) {
      vm->retval( new CoreArray( vm ) );
      return;
   }

   int32 pos_start = (int32) (start == 0 ? 0 : start->asInteger());
   int32 pos_end = (int32) (end == 0 ? array->length() : end->asInteger());
   if ( pos_start > pos_end ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_scan_end ) ) ) );
      return;
   }

   CoreArray *target = new CoreArray( vm );

   if ( pos_start < 0 || pos_start >= (int32) array->length() || pos_end > (int32) array->length()) {
       vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   Item *elements = array->elements();
   // fetching as we're going to change the stack
   Item func = *func_x;
   for( int32 i = pos_start; i < pos_end; i++ ) {
      vm->pushParameter( elements[i] );
      vm->callItemAtomic( func, 1 );
      if ( vm->regA().isTrue() ) {
         target->append( elements[i] );
      }
   }

   vm->retval( target );
}

/*#
   @function arrayMap
   @brief Maps an array using a given function.
   @param array The array to be mapped.
   @param mapFunc A function to map the array
   @optparam start Optional first element to be filtered.
   @optparam end Optional last element +1 to be filtered.
   @return An array built mapping each item in @b array.

   This function perform what is usually defined a “mapping” of the source array.
   The @b mapFunc parameter is a user-provide callable item
   that will be called once for each item in the specified array range;
   The value returned by the function will be appended in the returned array.

   A start-end range may optionally be specified to limit the mapping to a
   part of the array. As for other functions in this module, the range will
   cause mapping for the items from start to end-1.

   The mapping function is called in atomic mode. The called function cannot be
   interrupted by external kind requests, and it cannot sleep or yield the
   execution to other coroutines.

   @note This function is superseeded by @a map and may be removed
   in future versions.
*/
FALCON_FUNC  arrayMap( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *func_x = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_array_first ) ) ) );
      return;
   }

   if ( func_x == 0 || ! func_x->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_second_call ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_arrpar3 ) ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   // can find nothing in an empty array
   if ( array->length() == 0 ) {
      vm->retval( new CoreArray( vm ) );
      return;
   }

   int32 pos_start = (int32) (start == 0 ? 0 : start->asInteger());
   int32 pos_end = (int32) (end == 0 ? array->length() : end->asInteger());
   if ( pos_start > pos_end ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_scan_end ) ) ) );
      return;
   }

   CoreArray *target = new CoreArray( vm );

   if ( pos_start < 0 || pos_start >= (int32) array->length() || pos_end > (int32) array->length()) {
       vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   Item *elements = array->elements();
   // fetching as we're going to change the stack
   Item func = *func_x;
   for( int32 i = pos_start; i < pos_end; i++ ) {
      vm->pushParameter( elements[i] );
      vm->callItemAtomic( func, 1 );
      if ( !vm->regA().isOob() ) {
         target->append( vm->regA() );
      }
   }

   vm->retval( target );
}

#define MINMAL_QUICKSORT_THRESHOLD     4
inline void arraySort_swap( Item *a, int i, int j)
{
   Item T = a[i];
   a[i] = a[j];
   a[j] = T;
}

static void arraySort_quickSort( Item *array, int32 l, int32 r )
{
   int32 i;
   int32 j;
   Item v;

   if ( ( r-l ) > MINMAL_QUICKSORT_THRESHOLD )
   {
      i = ( r + l ) / 2;
      if ( array[l].compare( array[i] ) > 0 ) arraySort_swap( array, l , i );
      if ( array[l].compare( array[r] ) > 0 ) arraySort_swap( array , l, r );
      if ( array[i].compare( array[r] ) > 0 ) arraySort_swap( array, i, r );

      j = r - 1;
      arraySort_swap( array , i, j );
      i = l;
      v = array[j];
      for(;;)
      {
            while( array[++i].compare( v ) < 0 );
            while( array[--j].compare( v ) > 0 );
            if ( j < i ) break;
            arraySort_swap ( array , i , j );
      }
      arraySort_swap( array , i, r - 1 );
      arraySort_quickSort( array, l , j );
      arraySort_quickSort( array, i + 1 , r );
   }
}


static void arraySort_insertionSort( VMachine *vm, Item *array, int lo0, int hi0)
{
   int32 i;
   int32 j;
   Item v;


   for ( i = lo0 + 1 ; i <= hi0 ; i++ )
   {
      v = array[i];
      j = i;
      while ( ( j > lo0 ) && ( vm->compareItems( array[j-1], v )  > 0 ) )
      {
         array[j] = array[j-1];
         j--;
      }
      array[j] = v;
   }
}

static void arraySort_quickSort_flex( VMachine *vm, Item *comparer, Item *array, int32 l, int32 r )
{
   int32 i;
   int32 j;
   Item v;

   if ( ( r-l ) > MINMAL_QUICKSORT_THRESHOLD )
   {
      i = ( r + l ) / 2;

      vm->pushParameter( array[l] );
      vm->pushParameter( array[i] );
      vm->callItemAtomic( *comparer, 2 );

      if ( vm->regA().asInteger() > 0 ) arraySort_swap( array, l , i );

      vm->pushParameter( array[l] );
      vm->pushParameter( array[r] );
      vm->callItemAtomic( *comparer, 2 );

      if ( vm->regA().asInteger() > 0 ) arraySort_swap( array , l, r );

      vm->pushParameter( array[i] );
      vm->pushParameter( array[r] );
      vm->callItemAtomic( *comparer, 2 );
      if ( vm->regA().asInteger() > 0 ) arraySort_swap( array, i, r );

      j = r - 1;
      arraySort_swap( array, i, j );
      i = l;
      v = array[j];
      for(;;)
      {
            do {
               vm->pushParameter( array[++i] );
               vm->pushParameter( v );
               vm->callItemAtomic( *comparer, 2 );
            }
            while( vm->regA().asInteger() < 0 );

            do {
               vm->pushParameter( array[--j] );
               vm->pushParameter( v );
               vm->callItemAtomic( *comparer, 2 );
            }
            while( vm->regA().asInteger() > 0 );

            if ( j < i ) break;
            arraySort_swap ( array , i , j );
      }
      arraySort_swap( array , i, r - 1 );
      arraySort_quickSort( array, l , j );
      arraySort_quickSort( array, i + 1 , r );
   }
}


static void arraySort_insertionSort_flex( VMachine *vm, Item *comparer, Item *array, int lo0, int hi0)
{
   int32 i;
   int32 j;
   Item v;

   for ( i = lo0 + 1 ; i <= hi0 ; i++ )
   {
      v = array[i];
      j = i;
      while ( j > lo0 )
      {
         vm->pushParameter( array[j-1] );
         vm->pushParameter( v );
         vm->callItemAtomic( *comparer, 2 );

         if ( vm->regA().asInteger() <= 0 )
            break;

         array[j] = array[j-1];
         j--;
      }
      array[j] = v;
   }
}


/*#
   @function arraySort
   @brief Sorts an array, possibly using an arbitrary ordering criterion.
   @param array The array that will be searched.
   @optparam sortingFunc A function used to compare two items.
   @optparam start Optional first element to be searched.
   @optparam end Optional upper end of the range..

   The function sorts the contents of the array so that the first element is the
   smaller one, and the last element is the bigger one. String sorting is performed
   lexicographically. To sort the data based on an arbitrary criterion, or to sort
   complex items, or objects, based on some of their contents, the caller may
   provide a sortFunc that will receive two parameters. The sortFunc must return -1
   if the first parameter is to be considered smaller than the second, 0 if they
   are the same and 1 if the second parameter is considered greater.

   Sort function is called in atomic mode. The called function cannot be
   interrupted by external kind requests, and it cannot sleep or yield the
   execution to other coroutines.
*/

FALCON_FUNC  arraySort( ::Falcon::VMachine *vm )
{
   Item *array_itm = vm->param( 0 );
   Item *sorter_itm = vm->param( 1 );

   if ( array_itm == 0 || ! array_itm->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_array_first ) ) ) );
      return;
   }

   if ( sorter_itm != 0 && ! sorter_itm->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( rtl_second_call ) ) ) );
      return;
   }

   CoreArray *array = array_itm->asArray();
   Item *vector = array->elements();

   if ( sorter_itm == 0 ) {
      arraySort_quickSort( vector, 0, array->length() - 1 );
      arraySort_insertionSort( vm, vector, 0, array->length() -1 );
   }
   else {
      // fetching as we're going to change the stack
      Item sorter = *sorter_itm;

      arraySort_quickSort_flex( vm, &sorter, vector, 0, array->length() - 1 );
      arraySort_insertionSort_flex( vm, &sorter, vector, 0, array->length() -1 );
   }
}

/*#
   @function arrayMerge
   @brief Merges two arrays.
   @param array1 Array containing the first half of the merge, that will be modified.
   @param array2 Array containing the second half of the merge, read-only
   @optparam insertPos Optional position of array 1 at which to place array2
   @optparam start First element of array2 to merge in array1
   @optparam end Last element – 1 of array2 to merge in array1

   The items in array2 are appended to the end of array1, or in case an mergePos
   is specified, they are inserted at the given position. If mergePos is 0, they
   are inserted at beginning of array1, while if it's equal to array1 size they are
   appended at the end. An optional start parameter may be used to specify the
   first element in the array2 that must be copied in array1; if given, the
   parameter end will specify the last element that must be copied plus 1; that is
   elements are copied from array2 in array1 from start to end excluded.

   The items are copied shallowly. This means that if an object is in array2 and
   it's modified thereafter, both array2 and array1 will grant access to the
   modified objec
*/

FALCON_FUNC  arrayMerge( ::Falcon::VMachine *vm )
{
   Item *first_i = vm->param( 0 );
   Item *second_i = vm->param( 1 );

   Item *from_i =  vm->param( 2 );
   Item *start_i =  vm->param( 3 );
   Item *end_i =  vm->param( 4 );

   if( first_i == 0 || ! first_i->isArray() || second_i == 0 || ! second_i->isArray() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_need_two_arr ) ) ) );
      return;
   }

   if ( ( from_i != 0 && ! from_i->isOrdinal() ) ||
        ( start_i != 0 && ! start_i->isOrdinal() ) ||
        ( end_i != 0 && ! end_i->isOrdinal() )  )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_idx_not_num ) ) ) );
      return;
   }

   CoreArray *first = first_i->asArray();
   CoreArray *second = second_i->asArray();

   int64 from = from_i == 0 ? first->length(): from_i->forceInteger();
   int64 start = start_i == 0 ? 0 : start_i->forceInteger();
   int64 end = end_i == 0 ? second->length() : end_i->forceInteger();

   bool val;
   if( start == 0 && end == second->length() )
   {
       val = first->change( *second, (int32) from, (int32) from );
   }
   else {
      //TODO: partition on an array
      CoreArray *third = second->partition( (int32)start, (int32)end );
      if ( third ==  0 ) {
         vm->raiseError( e_arracc );
         return;
      }
      val = first->change( *third, (int32) from, (int32) from );
   }

   if ( ! val ) {
       vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ).extra( FAL_STR( rtl_start_outrange ) ) ) );
      return;
   }
}

}}

/* end of array.cpp */
