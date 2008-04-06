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
   @file array.cpp  Array specialized operation
   
   Theese operations replicate VM array management,
   but they are more flexible and use atomic calls in
   callbacks.
*/

/*#
   @beginmodule falcon_rtl
*/

/*#
   @funset Arrays Array related functions.
   @brief Array related functions.
   @beginfunset Arrays
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/carray.h>
#include <falcon/vm.h>
#include "rtl_messages.h"

namespace Falcon { namespace Ext {

/*# 
   @function arrayIns
   @brief Inserts an item into an array.
   @param array The array where the item should be placed.
   @param itempos  The position where the item should be placed.
   @param item the item to be exchanged.

    TODO: Write this text
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
      vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) )
         );
   }
}

/**
   arrayDel( array, element_to_be_deleted )
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
         vm->retval(1);
         return;
      }
   }

   vm->retval(0);
}

/**
   arrayDelAll( array, element_to_be_deleted )
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

   vm->retval( done ? 1 : 0 );
}


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

/**
   arrayRemove( array, start, end )  [remove in place]
   arrayRemove( array, itemPos )  [remove in place]
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
      vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
   }
}

/**
   arrayCopy( array, [start, end] ) --> copy  (equivalent to array[start:end] )
*/
FALCON_FUNC  arrayCopy( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item_start = vm->param(1);
   Item *item_end = vm->param(2);

   if ( array_x == 0 || !array_x->isArray() ||
         (item_start != 0 && !item_start->isOrdinal()) ||
         (item_end != 0 && !item_end->isOrdinal()) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_array_missing ) ) ) );
      return;
   }

   CoreArray *array = array_x->asArray();
   int64 start = item_start == 0 ? 0 : item_start->forceInteger();
   int64 end = item_end == 0 ? array->length() : item_end->forceInteger();
   CoreArray *arr1 = array->partition( (int32) start, (int32) end );
   if ( arr1 == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_array_missing ) ) ) );
      return;
   }
   vm->retval( arr1 );
}


FALCON_FUNC  arrayBuffer ( ::Falcon::VMachine *vm )
{
   Item *item_size = vm->param(0);

   if ( item_size == 0 || ! item_size->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   int32 nsize = (int32) item_size->forceInteger();
   CoreArray *array = new CoreArray( vm, nsize );
   Item *mem = array->elements();

   for ( int i = 0; i < nsize; i++ ) {
      mem[i].setNil();
   }
   array->length( nsize );
   vm->retval( array );
}

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
      vm->raiseModError( new RangeError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_emptyarr ) ) ) );
      return;
   }

   vm->retval( array->at(0) );
   array->remove(0);
}


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
      vm->raiseModError( new RangeError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_emptyarr ) ) ) );
      return;
   }

   vm->retval( array->at(array->length()-1) );
   array->remove(array->length()-1);
}


FALCON_FUNC  arrayFind ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *item = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() || item == 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar1 ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar3 ) ) ) );
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
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_scan_end ) ) ) );
      return;
   }

   if ( pos_start < 0 || pos_start >= (int32) array->length() || pos_end > (int32) array->length()) {
      vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
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


FALCON_FUNC  arrayScan ( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *func_x = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_array_first ) ) ) );
      return;
   }

   if ( func_x == 0 || ! func_x->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_second_call ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar3 ) ) ) );
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
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_scan_end ) ) ) );
      return;
   }

   if ( pos_start < 0 || pos_start >= (int32) array->length() ||
         pos_end > (int32) array->length()) {
       vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
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


FALCON_FUNC  arrayFilter( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *func_x = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_array_first ) ) ) );
      return;
   }

   if ( func_x == 0 || ! func_x->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_second_call ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar3 ) ) ) );
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
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_scan_end ) ) ) );
      return;
   }

   CoreArray *target = new CoreArray( vm );

   if ( pos_start < 0 || pos_start >= (int32) array->length() || pos_end > (int32) array->length()) {
       vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
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


FALCON_FUNC  arrayMap( ::Falcon::VMachine *vm )
{
   Item *array_x = vm->param(0);
   Item *func_x = vm->param(1);
   Item *start = vm->param(2);
   Item *end = vm->param(3);

   if ( array_x == 0 || ! array_x->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_array_first ) ) ) );
      return;
   }

   if ( func_x == 0 || ! func_x->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_second_call ) ) ) );
      return;
   }

   if ( start != 0 && ! start->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar2 ) ) ) );
      return;
   }

   if ( end != 0 && ! end->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_arrpar3 ) ) ) );
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
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_scan_end ) ) ) );
      return;
   }

   CoreArray *target = new CoreArray( vm );

   if ( pos_start < 0 || pos_start >= (int32) array->length() || pos_end > (int32) array->length()) {
       vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
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


/**
   arraySort( array ) -- > nil (in place)
   arraySort( array, callable ) -- > nil (in place)
*/
FALCON_FUNC  arraySort( ::Falcon::VMachine *vm )
{
   Item *array_itm = vm->param( 0 );
   Item *sorter_itm = vm->param( 1 );

   if ( array_itm == 0 || ! array_itm->isArray() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_array_first ) ) ) );
      return;
   }

   if ( sorter_itm != 0 && ! sorter_itm->isCallable() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_second_call ) ) ) );
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
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_need_two_arr ) ) ) );
      return;
   }

   if ( ( from_i != 0 && ! from_i->isOrdinal() ) ||
        ( start_i != 0 && ! start_i->isOrdinal() ) ||
        ( end_i != 0 && ! end_i->isOrdinal() )  )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_idx_not_num ) ) ) );
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
       vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_start_outrange ) ) ) );
      return;
   }
}

}}

/* end of array.cpp */
