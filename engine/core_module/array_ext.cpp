/*
   FALCON - The Falcon Programming Language.
   FILE: array_ext.cpp

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
#include <falcon/coretable.h>
#include <falcon/vm.h>
#include <falcon/eng_messages.h>


#include <string.h>

namespace Falcon {
namespace core {

/*#
   @method comp Array
   @brief Appends elements to this array through a filter.
   @param source A sequence, a range or a callable generating items.
   @optparam filter A filtering function receiving one item at a time.
   @return This array.

   Please, see the description of @a Sequence.comp.
   @see Sequence.comp
*/
FALCON_FUNC  Array_comp ( ::Falcon::VMachine *vm )
{
   if ( vm->param(0) == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "R|A|C|Sequence, [C]" ) );
   }

   // Save the parameters as the stack may change greatly.
   CoreArray* arr = vm->self().asArray();

   Item i_gen = *vm->param(0);
   Item i_check = vm->param(1) == 0 ? Item(): *vm->param(1);

   arr->items().comprehension_start( vm, vm->self(), i_check );
   vm->pushParam( i_gen );

}

/*#
   @method mcomp Array
   @brief Appends multiple elements to this array.
   @param ... One or more sequences or item generators.
   @return This array.

   Please, see the description of @a Sequence.mcomp.
   @see Sequence.mcomp
*/
FALCON_FUNC  Array_mcomp ( ::Falcon::VMachine *vm )
{
   CoreArray* arr = vm->self().asArray();
   StackFrame* current = vm->currentFrame();
   arr->items().comprehension_start( vm, vm->self(), Item() );

   for( uint32 i = 0; i < current->m_param_count; ++i )
   {
      vm->pushParam( current->m_params[i] );
   }
}

/*#
   @method mfcomp Array
   @brief Appends multiple elements to this array through a filter.
   @param filter A filter function receiving each element before its insertion, or nil.
   @param ... One or more sequences or item generators.
   @return This array.

   Please, see the description of @a Sequence.mfcomp.
   @see Sequence.mfcomp
*/
FALCON_FUNC  Array_mfcomp ( ::Falcon::VMachine *vm )
{
   if ( vm->param(0) == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "C, ..." ) );
   }

   CoreArray* arr = vm->self().asArray();
   Item i_check = *vm->param(0);
   StackFrame* current = vm->currentFrame();
   arr->items().comprehension_start( vm, vm->self(), i_check );

   for( uint32 i = 1; i < current->m_param_count; ++i )
   {
      vm->pushParam( current->m_params[i] );
   }
}


/*#
   @method front Array
   @brief Returns and eventually extracts the first element in the array.
   @optparam remove true to remove the front item.
   @return The extracted item.
   @raise AccessError if the array is empty.
*/
FALCON_FUNC  Array_front ( ::Falcon::VMachine *vm )
{

   CoreArray *array = vm->self().asArray();

   if ( array->length() == 0 )
   {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->moduleString( rtl_emptyarr ) ) );
      return;
   }

   vm->retval( array->at(0) );
   if ( vm->param(0) != 0 && vm->param(0)->isTrue() )
      array->remove(0);
}

/*#
   @method back Array
   @brief Returns and eventually extracts the last element in the array.
   @optparam remove true to remove the front item.
   @return The extracted item.
   @raise AccessError if the array is empty.
*/
FALCON_FUNC  Array_back ( ::Falcon::VMachine *vm )
{

   CoreArray *array = vm->self().asArray();

   if ( array->length() == 0 )
   {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_emptyarr ) ) );
   }

   vm->retval( array->at( array->length() - 1 ) );
   if ( vm->param(0) != 0 && vm->param(0)->isTrue() )
      array->remove( array->length() - 1 );
}

/*#
   @method table Array
   @brief Returns the table related with this array.
   @raise AccessError if the item is not an array.
   @return The table of which this item is a row.

   This array method retreives the table that is related with the
   item, provided the item is an array being part of a table.

   In case the item is an array, but it doesn't belong to any
   table, nil is returned.
*/

FALCON_FUNC Array_table( VMachine *vm )
{
   CoreArray *array = vm->self().asArray();
   if ( array->table() != 0 )
   {
      vm->retval( array->table() );
      return;
   }

   vm->retnil();
}

/*#
   @method tabField Array
   @brief Returns one of the items in the array, given the field name.
   @param field The field name or position to be retreived.
   @raise AccessError if the item is not an array.
   @return An item in the array or the default column value.

   If this item is an array and is part of a table, the field with
   the given name or ID (number) is searched in the table definition,
   and if found, it is returned. If the coresponding item in the array
   is nil, then the table column data (default data) is returned instead,
   unless the item is also an OOB item. In that case, nil is returned
   and the default column value is ignored.
*/

FALCON_FUNC Array_tabField( VMachine *vm )
{
   Item *i_field = vm->param( 0 );
   if ( i_field == 0 ||
      ! ( i_field->isString() || i_field->isOrdinal() ))
   {
      throw new ParamError( ErrorParam( e_inv_params ).
               extra( "(N)" ) );
      return;
   }

   CoreArray *array = vm->self().asArray();
   if ( array->table() != 0 )
   {
      // if a field paramter is given, return its value
      uint32 num = (uint32) CoreTable::noitem;
      CoreTable *table = reinterpret_cast<CoreTable*>(array->table()->getFalconData());

      if ( i_field->isString() )
      {
         num = table->getHeaderPos( *i_field->asString() );
      }
      else {
         // we already checked, must be a field
         num = (uint32) i_field->forceInteger();
      }

      if ( num < array->length() )
      {
         Item &data = (*array)[num];
         if ( ! data.isNil() )
         {
            vm->retval( data );
         }
         else {
            if ( data.isOob() )
               vm->retnil();
            else
               vm->retval( table->columnData( num ) );
         }

         return;
      }

      throw new AccessError( ErrorParam( e_param_range ) );
      return;
   }

   vm->retnil();
}

/*#
   @method tabRow Array
   @brief Returns the row ID of this element.
   @raise AccessError if the item is not a table row (array).
   @return A number indicating the position of this row in the table, or
      nil if this item is an array, but it's not stored in a table.

   This method returns the position of this element in a table.

   This number gets valorized only after a @a Table.get or @a Table.find
   method call, so that it is possible to know what index had this element
   in the owning table. If the table is changed by inserting or removing
   a row, the number returned by this function becomes meaningless.
*/

FALCON_FUNC Array_tabRow( VMachine *vm )
{
   CoreArray *array = vm->self().asArray();
   if ( array->table() != 0 )
      vm->retval( (int64) array->tablePos() );
   else
      vm->retnil();
}

/*#
   @method first Array
   @brief Returns an iterator to the head of this array.
   @return An iterator.
*/

FALCON_FUNC Array_first( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   // we need to set the FalconData flag
   iterator->setUserData( new Iterator( &vm->self().asArray()->items() ) );
   vm->retval( iterator );
}

/*#
   @method last Array
   @brief Returns an iterator to the last element in this array.
   @return An iterator.
*/

FALCON_FUNC Array_last( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   // we need to set the FalconData flag
   iterator->setUserData( new Iterator( &vm->self().asArray()->items(), true ) );
   vm->retval( iterator );
}

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

/*#
   @method ins Array
   @brief Inserts an item into this array array.
   @param itempos  The position where the item should be placed.
   @param item The item to be inserted.

   The item is inserted before the given position. If pos is 0, the item is
   inserted in the very first position, while if it's equal to the array length, it
   is appended at the array tail.
*/

FALCON_FUNC  mth_arrayIns ( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *item_pos;
   Item *item;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      item_pos = vm->param(0);
      item = vm->param(1);
   }
   else
   {
      array_x = vm->param(0);
      item_pos = vm->param(1);
      item = vm->param(2);
   }


   if ( array_x == 0 || ! array_x->isArray() ||
         item_pos == 0 || ! item_pos->isOrdinal() ||
         item == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->self().isMethodic() ? "N,X" : "A,N,X" ) );
   }

   CoreArray *array = array_x->asArray();
   int32 pos = (int32) item_pos->forceInteger();

   if ( ! array->insert( *item, pos ) ) {
      // array access error
      throw new AccessError( ErrorParam( e_arracc, __LINE__ )
         .origin( e_orig_runtime ) );
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

/*#
   @method del Array
   @brief Deletes the first element matching a given item.
   @param item The item that must be deleted.
   @return true if at least one item has been removed, false otherwise.

   The function scans the array searching for an item that is considered
   equal to the given one. If such an item can be found, it is removed and
   the function returns true. If the item cannot be found, false is returned.

   Only the first item matching the given one will be deleted.
*/
FALCON_FUNC  mth_arrayDel ( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *item_rem;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      item_rem = vm->param(0);
   }
   else
   {
      array_x = vm->param(0);
      item_rem = vm->param(1);
   }

   if ( array_x == 0 || ! array_x->isArray()
         || item_rem == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "X" : "A,X" ) );
   }

   // find the element
   CoreArray *array = array_x->asArray();
   const ItemArray& elements = array->items();
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

/*#
   @method delAll Array
   @brief Deletes all the occurrences of a given item in this array.
   @param item The item that must be deleted.
   @return true if at least one item has been removed, false otherwise.

   This function removes all the elements in this array that are
   considered equivalent to the given item. If one or more elements have
   been found and deleted, the function will return true, else it will
   return false.

   @see filter
*/
FALCON_FUNC  mth_arrayDelAll ( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *item_rem;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      item_rem = vm->param(0);
   }
   else
   {
      array_x = vm->param(0);
      item_rem = vm->param(1);
   }

   if ( array_x == 0 || ! array_x->isArray()
       || item_rem == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "X" : "A,X" ) );
   }

   // find the element
   bool done = false;
   CoreArray *array = array_x->asArray();
   const ItemArray& elements = array->items();
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
   @param item The item to be added.

   The element will be added at the end of the array,
   and its size will be increased by one.
*/

/*#
   @method add Array
   @brief Adds an element to this array.
   @param item The item to be added.

   The element will be added at the end of this array,
   and its size will be increased by one.
*/
FALCON_FUNC  mth_arrayAdd ( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *item;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      item = vm->param(0);
   }
   else
   {
      array_x = vm->param(0);
      item = vm->param(1);
   }

   if ( array_x == 0 || ! array_x->isArray()
         || item == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "X" : "A,X" ) );
   }

   CoreArray *array = array_x->asArray();
   array->append( *item );
}

/*#
   @function arrayFill
   @brief Fills the array with the given element.
   @param array The array where to add the new item.
   @param item The item to be replicated.
   @return The same @b array passed as parameter.

   This method allows to clear a whole array, resetting all the elements
   to a default value.
*/

/*#
   @method fill Array
   @brief Fills the array with the given element.
   @param item The item to be replicated.
   @return This array.

   This method allows to clear a whole array, resetting all the elements
   to a default value.
*/
FALCON_FUNC  mth_arrayFill ( ::Falcon::VMachine *vm )
{
   Item *i_array;
   Item *i_item;

   if ( vm->self().isMethodic() )
   {
      i_array = &vm->self();
      i_item = vm->param(0);
   }
   else
   {
      i_array = vm->param(0);
      i_item = vm->param(1);
   }

   if ( i_array == 0 || ! i_array->isArray()
         || i_item == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "X" : "A,X" ) );
   }

   CoreArray *array = i_array->asArray();
   for ( uint32 i = 0; i < array->length(); i ++ )
   {

      if ( i_item->isString() )
         (*array)[i] = new CoreString( *i_item->asString() );
      else
         (*array)[i] = *i_item;
   }

   vm->retval( array );
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

FALCON_FUNC  mth_arrayResize ( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *item_size;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      item_size = vm->param(0);
   }
   else
   {
      array_x = vm->param(0);
      item_size = vm->param(1);
   }


   if ( array_x == 0 || array_x->type() != FLC_ITEM_ARRAY ||
         item_size == 0 || !item_size->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "N" : "A,N" ) );
   }

   CoreArray *array = array_x->asArray();
   int64 size = item_size->forceInteger();
   if ( size < 0 )
      size = 0;
   array->resize( (uint32) size );
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

/*#
   @method remove Array
   @brief Removes one or more elements in the array.
   @param itemPos The position of the item to be removed, or the first of the items to be removed.
   @optparam lastItemPos The last item to be removed, in range semantic.

   Remove one item or a range of items. The size of the array is shortened accordingly.
   The semantic of @b lastItemPos is the same as ranged access to the array.
*/

FALCON_FUNC  mth_arrayRemove( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *item_start;
   Item *item_end;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      item_start = vm->param(0);
      item_end = vm->param(1);
   }
   else
   {
      array_x = vm->param(0);
      item_start = vm->param(1);
      item_end = vm->param(2);
   }


   if ( array_x == 0 || !array_x->isArray() ||
         item_start == 0 || !item_start->isOrdinal() ||
         (item_end != 0 && !item_end->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "N,[N]" : "A,N,[N]" ) );
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
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) );
   }
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N,[X]" ) );
   }

   uint32 nsize = (uint32) item_size->forceInteger();

   CoreArray *array = new CoreArray;
   array->resize( nsize );

   if( i_item != 0 )
   {
      for ( uint32 i = 0; i < nsize; i++ ) {
         array->items()[i] = *i_item;
      }
   }

   vm->retval( array );
}

/*#
   @function arrayCompact
   @brief Reduces the memory used by an array.
   @param array The array itself.
   @return Itself

   Normally, array operations, as insertions, additions and so on
   cause the array to grow to accommodate items that may come in the future.
   also, reducing the size of an array doesn't automatically dispose of the
   memory held by the array to store its elements.

   This function grants that the memory used by the array is strictly the
   memory needed to store its data.
*/

/*#
   @method compact Array
   @brief Reduces the memory used by an array.
   @return Itself

   Normally, array operations, as insertions, additions and so on
   cause the array to grow to accommodate items that may come in the future.
   also, reducing the size of an array doesn't automatically dispose of the
   memory held by the array to store its elements.

   This method grants that the memory used by the array is strictly the
   memory needed to store its data.
*/
FALCON_FUNC  mth_arrayCompact ( ::Falcon::VMachine *vm )
{
   Item *array_x;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
   }
   else
   {
      array_x = vm->param(0);
      if ( array_x == 0 || !array_x->isArray() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "A" ) );
      }
   }

   array_x->asArray()->items().compact();
   vm->retval( *array_x );
}

/*#
   @function arrayHead
   @brief Extracts the first element of an array and returns it.
   @param array The array that will be modified.
   @return The extracted item.
   @raise AccessError if the array is empty.

   This function removes the first item of the array and returns it.
   If the original array is empty, AccessError is raised.

   @see Array.front
   @see Array.head
*/

/*#
   @method head Array
   @brief Extracts the first element from this array and returns it.
   @return The extracted item.
   @raise AccessError if the array is empty.

   This function removes the first item from the array and returns it.
   If the original array is empty, AccessError is raised.

   @see Array.front
*/
FALCON_FUNC  mth_arrayHead ( ::Falcon::VMachine *vm )
{
   Item *array_x;

   if( vm->self().isMethodic() )
   {
      array_x = &vm->self();
   }
   else
   {
      array_x = vm->param(0);

      if ( array_x == 0 || ! array_x->isArray() ) {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            origin( e_orig_runtime ).extra( "A" ) );
      }
   }

   CoreArray *array = array_x->asArray();
   if ( array->length() == 0 )
   {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_emptyarr ) ) );
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

/*#
   @method tail Array
   @brief Extracts the last element of an array and returns it.
   @return The extracted item.

   This function removes the last item of the array and returns it.
   If the original array is empty, AccessError is raised.
*/

FALCON_FUNC  mth_arrayTail ( ::Falcon::VMachine *vm )
{
   Item *array_x;

   if( vm->self().isMethodic() )
   {
      array_x = &vm->self();
   }
   else
   {
      array_x = vm->param(0);

      if ( array_x == 0 || ! array_x->isArray() ) {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            origin( e_orig_runtime ).extra( "A" ) );
      }
   }

   CoreArray *array = array_x->asArray();
   if ( array->length() == 0 )
   {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_emptyarr ) ) );
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

/*#
   @method find Array
   @brief Searches for a given item in an array.
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
FALCON_FUNC  mth_arrayFind ( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *item;
   Item *start;
   Item *end;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      item = vm->param(0);
      start = vm->param(1);
      end = vm->param(2);
   }
   else
   {
      array_x = vm->param(0);
      item = vm->param(1);
      start = vm->param(2);
      end = vm->param(3);
   }

   if ( array_x == 0 || ! array_x->isArray()
        || item == 0
        || (start != 0 && ! start->isOrdinal())
        || ( end != 0 && ! end->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "X,[N],[N]" : "A,X,[N],[N]" ) );
   }

   CoreArray *array = array_x->asArray();
   // can find nothing in an empty array
   if ( array->length() == 0 ) {
      vm->retval( -1 );
      return;
   }

   int32 pos_start = (int32) (start == 0 ? 0 : start->asInteger());
   int32 pos_end = (int32) (end == 0 ? array->length() : end->asInteger());
   if ( pos_start > pos_end )
   {
      int32 temp = pos_start;
      pos_start = pos_end;
      pos_end = temp;
   }

   if ( pos_start < 0 || pos_start >= (int32) array->length() || pos_end > (int32) array->length()) {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) );
      return;
   }

   const ItemArray& elements = array->items();
   for( int32 i = pos_start; i < pos_end; i++ )
   {
      if ( elements[i] == *item )
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
   @param func Function that verifies the criterion.
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

/*#
   @method scan Array
   @brief Searches an array for an item satisfying arbitrary criteria.
   @param func Function that verifies the criterion.
   @optparam start Optional first element to be searched.
   @optparam end Optional upper end of the range..
   @return The position of the searched item in the array, or -1 if not found.

   @see arrayScan
*/
FALCON_FUNC  mth_arrayScan ( ::Falcon::VMachine *vm )
{
   Item *array_x;
   Item *func_x;
   Item *start;
   Item *end;

   if ( vm->self().isMethodic() )
   {
      array_x = &vm->self();
      func_x = vm->param(0);
      start = vm->param(1);
      end = vm->param(2);
   }
   else
   {
      array_x = vm->param(0);
      func_x = vm->param(1);
      start = vm->param(2);
      end = vm->param(3);
   }

   if ( array_x == 0 || ! array_x->isArray()
        || func_x == 0 || ! func_x->isCallable()
        || (start != 0 && ! start->isOrdinal())
        || ( end != 0 && ! end->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "C,[N],[N]" : "A,C,[N],[N]" ) );
   }

   CoreArray *array = array_x->asArray();
   // can find nothing in an empty array
   if ( array->length() == 0 ) {
      vm->retval( -1 );
      return;
   }

   int32 pos_start = (int32) (start == 0 ? 0 : start->asInteger());
   int32 pos_end = (int32) (end == 0 ? array->length() : end->asInteger());
   if ( pos_start > pos_end )
   {
      int32 temp = pos_start;
      pos_start = pos_end;
      pos_end = temp;
   }

   if ( pos_start < 0 || pos_start >= (int32) array->length() ||
         pos_end > (int32) array->length()) {
       throw new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) );
   }

   const ItemArray& elements = array->items();
   // fetching as we're going to change the stack
   Item func = *func_x;
   for( int32 i = pos_start; i < pos_end; i++ )
   {
      vm->pushParam( elements[i] );
      vm->callItemAtomic( func, 1 );
      if ( vm->regA().isTrue() ) {
         vm->retval( i );
         return;
      }
   }

   vm->retval(-1);
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
      if ( array[l] > array[i] ) arraySort_swap( array, l , i );
      if ( array[l] > array[r] ) arraySort_swap( array , l, r );
      if ( array[i] > array[r] ) arraySort_swap( array, i, r );

      j = r - 1;
      arraySort_swap( array , i, j );
      i = l;
      v = array[j];
      for(;;)
      {
            while( array[++i] < v );
            while( array[--j] > v );
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
      while ( ( j > lo0 ) && ( array[j-1] > v ) )
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

      vm->pushParam( array[l] );
      vm->pushParam( array[i] );
      vm->callItemAtomic( *comparer, 2 );

      if ( vm->regA().asInteger() > 0 ) arraySort_swap( array, l , i );

      vm->pushParam( array[l] );
      vm->pushParam( array[r] );
      vm->callItemAtomic( *comparer, 2 );

      if ( vm->regA().asInteger() > 0 ) arraySort_swap( array , l, r );

      vm->pushParam( array[i] );
      vm->pushParam( array[r] );
      vm->callItemAtomic( *comparer, 2 );
      if ( vm->regA().asInteger() > 0 ) arraySort_swap( array, i, r );

      j = r - 1;
      arraySort_swap( array, i, j );
      i = l;
      v = array[j];
      for(;;)
      {
            do {
               vm->pushParam( array[++i] );
               vm->pushParam( v );
               vm->callItemAtomic( *comparer, 2 );
            }
            while( vm->regA().asInteger() < 0 );

            do {
               vm->pushParam( array[--j] );
               vm->pushParam( v );
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
         vm->pushParam( array[j-1] );
         vm->pushParam( v );
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

/*#
   @method sort Array
   @brief Sorts an array, possibly using an arbitrary ordering criterion.
   @optparam sortingFunc A function used to compare two items.

   @see arraySort
*/
FALCON_FUNC  mth_arraySort( ::Falcon::VMachine *vm )
{
   Item *array_itm;
   Item *sorter_itm;

   if( vm->self().isMethodic() )
   {
      array_itm = &vm->self();
      sorter_itm = vm->param( 0 );
   }
   else
   {
      array_itm = vm->param( 0 );
      sorter_itm = vm->param( 1 );
   }

   if ( array_itm == 0 || ! array_itm->isArray()
        || (sorter_itm != 0 && ! sorter_itm->isCallable())
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->self().isMethodic() ? "C" : "A,C" ) );
   }

   CoreArray *array = array_itm->asArray();
   Item *vector = array->items().elements();

   if ( sorter_itm == 0 )
   {
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
   modified object.
*/

/*#
   @method merge Array
   @brief Merges the given array into this one.
   @param array Array containing the second half of the merge, read-only
   @optparam insertPos Optional position of array 1 at which to place array2
   @optparam start First element of array to merge in this array
   @optparam end Last element - 1 of array2 to merge in this array

   @see arrayMerge
*/

FALCON_FUNC  mth_arrayMerge( ::Falcon::VMachine *vm )
{
   Item *first_i, *second_i, *from_i, *start_i, *end_i;

   if( vm->self().isMethodic() )
   {
      first_i = &vm->self();
      second_i = vm->param( 0 );
      from_i =  vm->param( 1 );
      start_i =  vm->param( 2 );
      end_i =  vm->param( 3 );
   }
   else
   {
      first_i = vm->param( 0 );
      second_i = vm->param( 1 );
      from_i =  vm->param( 2 );
      start_i =  vm->param( 3 );
      end_i =  vm->param( 4 );
   }

   if ( first_i == 0 || ! first_i->isArray()
       || second_i == 0 || ! second_i->isArray()
       || ( from_i != 0 && ! from_i->isOrdinal() ) ||
        ( start_i != 0 && ! start_i->isOrdinal() ) ||
        ( end_i != 0 && ! end_i->isOrdinal() )  )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( vm->self().isMethodic() ? "A,[N],[N],[N]" : "A,A,[N],[N],[N]" ) );
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
      third->mark( vm->generation() );
      if ( third ==  0 ) {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) );
      }
      val = first->change( *third, (int32) from, (int32) from );
   }

   if ( ! val ) {
       throw new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ).extra( Engine::getMessage( rtl_start_outrange ) ) );
   }
}

/*#
   @function arrayNM
   @brief Force an array to be non-method.
   @param array the array that should not be transformed into a method.
   @return The same array passed as parameter.

   Callable arrays stored in objects become methods when they are accessed.
   At times, this is not desirable. Suppose it is necessary to store a list
   of items, possibly functions, in an object, and that random access is needed.

   In this case, an array is a good storage, but it's useful to tell Falcon
   that this array is never to be considered a method for the host object.
   Consider the following case:

   @code
   class WithArrays( a )
      asMethod = a
      asArray= arrayNM( a.clone() )  // or a.clone().NM()
   end

   wa = WithArrays( [ printl, 'hello' ] )

   // this is ok
   for item in wa.asArray: > "Item in this array: ", item

   // this will raise an error
   for item in wa.asMethod: > "Item in this array: ", item

   @endcode

   The array is cloned in this example as the function modifies
   the array itself, which becomes non-methodic, and returns it
   without copying it. So, if not copied, in this example also
   asMethod would have been non-methodic.
*/

/*#
   @method NM Array
   @brief Force an array to be non-method.
   @return The This array

   @see arrayNM
*/

FALCON_FUNC  mth_arrayNM( ::Falcon::VMachine *vm )
{
   Item *i_array;

   if( vm->self().isMethodic() )
   {
      i_array = &vm->self();
   }
   else
   {
      i_array = vm->param(0);

      if ( i_array == 0 || ! i_array->isArray() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "A" ) );
      }
   }


   CoreArray *array = i_array->asArray();
   array->canBeMethod(false);
   vm->retval( array );
}


}}

/* end of array_ext.cpp */
