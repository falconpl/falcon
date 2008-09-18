/*
   FALCON - The Falcon Programming Language
   FILE: funcext_ext.cpp

   Compiler module main file - extension implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Funcext module main file - extension implementation.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/lineardict.h>

#include "funcext_ext.h"
//#include "compiler_st.h"

#include <math.h>
#include <errno.h>


/*#
   @beginmodule feather_funcext
*/

namespace Falcon {

namespace Ext {

/*#
   @function at
   @brief Retreives or sets the nth element of a indexed sequence.
   @param sequence An array, string or another sequence.
   @param itempos A number or a range to access the sequence.
   @optparam item If given, will substitue the given item or range.

   The item is inserted before the given position. If pos is 0, the item is
   inserted in the very first position, while if it's equal to the array length, it
   is appended at the array tail.
*/

FALCON_FUNC  fe_at ( ::Falcon::VMachine *vm )
{
   Item *i_array = vm->param(0);
   Item *i_pos = vm->pam(1);
   Item *i_val = vm->param(2);

   CoreObject *self = 0;
   CoreClass *sourceClass=0;
   uint32 pos;

   if ( i_array == 0 || i_pos == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            origin( e_orig_runtime ).
            extra("A|S|C|O, N|R|S, [X]") ) );
      return
   }

   switch( i_array->type() )
   {
   case FLC_ITEM_ARRAY:
      CoreArray *ca = i_array->asArray();

      if ( i_pos->isOrdinal() )
      {
         int32 pos = i_pos->forceInteger();
         if ( pos < 0 ) pos = ca->length() - pos;
         if ( pos >= ca->length() )
         {
            vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
               origin( e_orig_runtime ) )
               );
            return;
         }
         vm->retval( ca->at( pos ) );
         if ( i_val != 0 )
            (*ca)[pos] = *i_val;

         return;
      }
      else if ( i_pos->isRange() )
      {
         int32 start = i_pos->asRangeStart();
         int32 end = i_pos->asRangeIsOpen() ? ca->length() : i_pos->asRangeEnd();
         CoreArray *part = ca->partition( start, end );

         if ( part != 0 )
         {
            vm->retval( part );

            if ( i_val != 0 )
            {
               if( i_val->isArray() ) {
                  if( ca->change( *i_val->asArray(), start, end ) )
                     return;
               }
               else
               {
                  if ( start != end )
                  {// insert
                     if( ca->remove( start, end ) )
                     {
                        if( ca->insert( *i_val, start ) )
                           return;
                     }
                  }
               }
            }
            else
               return;
         }
         else {
            vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
               origin( e_orig_runtime ) )
               );
            return;
         }
      }
   }
   break;

   case FLC_ITEM_STRING:
   {
      String *str = i_array->asString();

      if ( i_pos->isOrdinal() )
      {
         int32 pos = i_pos->forceInteger();
         if ( pos < 0 ) pos = str->length() - pos;
         if ( pos >= str->length() )
         {
            vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
               origin( e_orig_runtime ) )
               );
            return;
         }
         vm->retval( new GarbageString( vm, str->subString( pos, pos+1) ) );

         if ( i_val != 0 && i_val->isString() && i_val->asString()->size() > 0 )
         {
            str->setCharAt(pos, i_val->asString()->getCharAt(0) );
         }

         return;
      }
      else if ( i_pos->isRange() )
      {
         int32 start = i_pos->asRangeStart();
         int32 end = i_pos->asRangeIsOpen() ? str->length() : i_pos->asRangeEnd();

         vm->retval( new GarbageString( vm, str->subString( start, end ) ) );
         if ( i_val != 0 && i_val->isString() )
         {
            str->change( start, end, *i_val->asString() );
         }

         return;
      }
   }
   break;

   // dictionary?
   case FLC_ITEM_DICT:
      CoreDict *dict = i_array->asDict();
      if( i_pos != 0 )
      {
         if( i_val == 0 )
         {
            if ( dict->find( *i_pos, vm->regA() ) )
               return;
         }
         else {
            dict->insert( *i_pos, vm->regA() );
            return;
         }

         vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
            origin( e_orig_runtime ) )
            );
         return;
      }
      break;

   case FLC_ITEM_OBJECT:
      if ( i_pos->isString() )
      {
         Item prop;
         if( source->asObject()->getProperty( *i_pos->asString, prop ) )
         {
            // we must create a method if the property is a function.
            Item *p = prop.dereference();

            switch( p->type() ) {
               case FLC_ITEM_FUNC:
                  // the function may be a dead function; by so, the method will become a dead method,
                  // and it's ok for us.
                  vm->regA().setMethod( source->asObject(), p->asFunction(), p->asModule() );
                  break;

               case FLC_ITEM_CLASS:
                  vm->regA().setClassMethod( source->asObject(), p->asClass() );
                  break;

               default:
                  vm->regA() = *p;
            }
            //it's ok anyhow
            return;
         }

         vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
            origin( e_orig_runtime ) )
            );
         return;
      }
      break;

   case FLC_ITEM_CLSMETHOD:
         sourceClass = source->asMethodClass();
         self = source->asMethodObject();

   // do not break: fallback
   case FLC_ITEM_CLASS:
      if ( sourceClass == 0 )
         sourceClass = source->asClass();

      if( i_pos->isString() )
      {
         if( sourceClass->properties().findKey( *i_pos->asString(), pos ) )
         {
            Item *prop = sourceClass->properties().getValue( pos );

            // now, accessing a method in a class means that we want to call the base method in a
            // self item:
            if( prop->type() == FLC_ITEM_FUNC )
            {
               if ( self != 0 )
                  vm->regA().setMethod( self, prop->asFunction(), prop->asModule() );
               else
                  vm->regA().setFunction( prop->asFunction(), prop->asModule() );
            }
            else
            {
               vm->regA() = *prop;
            }
            return;
         }

         vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
            origin( e_orig_runtime ) )
            );

         return;
      }
      break;
   }

   vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
      origin( e_orig_runtime ).
      extra("A,N|R,[X]") ) );
}


/*#
   @function gt
   @brief Performs a lexicographic check for the first operand being greater than the second.
   @param a First operand
   @param b Second operand
   @return true if a > b, false otherwise.
*/
FALCON_FUNC  fe_gt ( ::Falcon::VMachine *vm )
{
   Item *i_a = vm->param(0);
   Item *i_b = vm->param(1);

   if ( i_a == 0 || i_b == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,X") ) );
      return;
   }

   vm->regA().setBoolean( vm->compareItems( *i_a, *i_b ) > 0 );
}

/*#
   @function ge
   @brief Performs a lexicographic check for the first operand being greater or equal to the second.
   @param a First operand
   @param b Second operand
   @return true if a >= b, false otherwise.
*/
FALCON_FUNC  fe_ge ( ::Falcon::VMachine *vm )
{
   Item *i_a = vm->param(0);
   Item *i_b = vm->param(1);

   if ( i_a == 0 || i_b == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,X") ) );
      return;
   }

   vm->regA().setBoolean( vm->compareItems( *i_a, *i_b ) >= 0 );
}

/*#
   @function lt
   @brief Performs a lexicographic check for the first operand being less than the second.
   @param a First operand
   @param b Second operand
   @return true if a < b, false otherwise.
*/
FALCON_FUNC  fe_lt ( ::Falcon::VMachine *vm )
{
   Item *i_a = vm->param(0);
   Item *i_b = vm->param(1);

   if ( i_a == 0 || i_b == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,X") ) );
      return;
   }

   vm->regA().setBoolean( vm->compareItems( *i_a, *i_b ) < 0 );
}

/*#
   @function le
   @brief Performs a lexicographic check for the first operand being less or equal to the second.
   @param a First operand
   @param b Second operand
   @return true if a <= b, false otherwise.
*/
FALCON_FUNC  fe_le ( ::Falcon::VMachine *vm )
{
   Item *i_a = vm->param(0);
   Item *i_b = vm->param(1);

   if ( i_a == 0 || i_b == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,X") ) );
      return;
   }

   vm->regA().setBoolean( vm->compareItems( *i_a, *i_b ) <= 0 );
}

/*#
   @function eq
   @brief Performs a lexicographic check for the first operand being equal to the second.
   @param a First operand
   @param b Second operand
   @return true if a == b, false otherwise.
*/
FALCON_FUNC  fe_eq ( ::Falcon::VMachine *vm )
{
   Item *i_a = vm->param(0);
   Item *i_b = vm->param(1);

   if ( i_a == 0 || i_b == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,X") ) );
      return;
   }

   vm->regA().setBoolean( vm->compareItems( *i_a, *i_b ) == 0 );
}

/*#
   @function neq
   @brief Performs a lexicographic check for the first operand being not equal to the second.
   @param a First operand
   @param b Second operand
   @return true if a == b, false otherwise.
*/
FALCON_FUNC  fe_neq ( ::Falcon::VMachine *vm )
{
   Item *i_a = vm->param(0);
   Item *i_b = vm->param(1);

   if ( i_a == 0 || i_b == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X,X") ) );
      return;
   }

   vm->regA().setBoolean( vm->compareItems( *i_a, *i_b ) != 0 );
}

/*#
   @function deq
   @brief Performs a deep equality check.
   @param a First operand
   @param b Second operand
   @return true if the two items are deeply equal (same content).

*/

static bool internal_eq( ::Falcon::VMachine *vm, const Item &first, const Item &second )
{
   if( first == second || vm->compareItems( first, second ) == 0 )
   {
      return true;
   }

   if( first.isArray() && second.isArray() )
   {
      CoreArray *arr1 = first.asArray();
      CoreArray *arr2 = second.asArray();

      if ( arr1->length() != arr2->length() )
         return false;

      for ( uint32 p = 0; p < arr1->length(); p++ )
      {
         if ( ! internal_eq( vm, arr1->at(p), arr2->at(p) ) )
            return false;
      }

      return true;
   }

   if( first.isDict() && second.isDict() )
   {
      CoreDict *d1 = first.asDict();
      CoreDict *d2 = second.asDict();

      if ( d1->length() != d2->length() )
         return false;

      DictIterator *di1 = d1->first();
      DictIterator *di2 = d2->first();
      while( di1->isValid() )
      {
         if ( ! internal_eq( vm, di1->getCurrentKey(), di2->getCurrentKey() ) ||
              ! internal_eq( vm, di1->getCurrent(), di2->getCurrent() ) )
         {
            delete d1;
            delete d2;
            return false;
         }
      }

      delete d1;
      delete d2;
      return true;
   }

   return false;
}


FALCON_FUNC  fe_deq( ::Falcon::VMachine *vm )
{
   Item *first = vm->param(0);
   Item *second = vm->param(1);
   if ( first == 0 || second == 0 )
   {
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( "X,X" ) ) );
      return;
   }

   vm->retval( internal_eq( vm, *first, *second ) ? 1:0);
}


/*#
   @function add
   @brief Adds two items along the rules of VM '+' operator.
   @param a First operand
   @param b Second operand
   @return The result of the sum.

   This function operates also on strings, arrays and dictionaries
   replicating the behavior of the "+" operator as it is performed
   by the VM.
*/

FALCON_FUNC  fe_add( ::Falcon::VMachine *vm )
{
   Item *operand1 = vm->param(0);
   Item *operand2 = vm->param(1);
   if ( operand2 == 0 || operand2 == 0 )
   {
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "X,X" ) ) );
      return;
   }

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         vm->regA().setInteger( operand1->asInteger() + operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         vm->regA().setNumeric( operand1->asInteger() + operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         vm->regA().setNumeric( operand1->asNumeric() + operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         vm->regA().setNumeric( operand1->asNumeric() + operand2->asNumeric() );
      return;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_INT:
      case FLC_ITEM_STRING<< 8 | FLC_ITEM_NUM:
      {
         int64 chr = operand2->forceInteger();
         if ( chr >= 0 && chr <= (int64) 0xFFFFFFFF )
         {
            GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
            gcs->append( (uint32) chr );
            vm->regA().setString( gcs );
            return;
         }
      }
      break;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_STRING:
      {
         GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
         gcs->append( *operand2->asString() );
         vm->regA().setString( gcs );
      }
      return;

      case FLC_ITEM_DICT<< 8 | FLC_ITEM_DICT:
      {
         CoreDict *dict = new LinearDict( vm, operand1->asDict()->length() + operand2->asDict()->length() );
         dict->merge( *operand1->asDict() );
         dict->merge( *operand2->asDict() );
         vm->regA().setDict( dict );
      }
      return;
   }

   // add any item to the end of an array.
   if( operand1->type() == FLC_ITEM_ARRAY )
   {
      CoreArray *first = operand1->asArray()->clone();

      if ( operand2->type() == FLC_ITEM_ARRAY ) {
         first->merge( *operand2->asArray() );
      }
      else {
         if ( operand2->isString() && operand2->asString()->garbageable() )
            first->append( operand2->asString()->clone() );
         else
            first->append( *operand2 );
      }
      vm->retval( first );
      return;
   }

   vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "X,X" ) ) );
}


/*#
   @function sub
   @brief Subtracts two items along the rules of VM '-' operator.
   @param a First operand
   @param b Second operand
   @return The result of the subtraction.

   This function operates also on strings, arrays and dictionaries
   replicating the behavior of the "-" operator as it is performed
   by the VM.
*/

FALCON_FUNC  fe_sub( ::Falcon::VMachine *vm )
{
   Item *operand1 = vm->param(0);
   Item *operand2 = vm->param(1);
   if ( operand2 == 0 || operand2 == 0 )
   {
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "X,X" ) ) );
      return;
   }

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         vm->regA().setInteger( operand1->asInteger() - operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         vm->regA().setNumeric( operand1->asInteger() - operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         vm->regA().setNumeric( operand1->asNumeric() - operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         vm->regA().setNumeric( operand1->asNumeric() - operand2->asNumeric() );
      return;
   }

   // remove any element from an array
   if ( operand1->isArray() )
   {
      CoreArray *source = operand1->asArray();
      CoreArray *dest = source->clone();

      // if we have an array, remove all of it
      if( operand2->isArray() )
      {
         CoreArray *removed = operand2->asArray();
         for( uint32 i = 0; i < removed->length(); i ++ )
         {
            int32 rem = dest->find( removed->at(i) );
            if( rem >= 0 )
               dest->remove( rem );
         }
      }
      else {
         int32 rem = dest->find( *operand2 );
         if( rem >= 0 )
            dest->remove( rem );
      }

      // never raise.
      vm->regA() = dest;
      return;
   }
   // remove various keys from arrays
   else if( operand1->isDict() )
   {
      CoreDict *source = operand1->asDict();
      CoreDict *dest = source->clone();

      // if we have an array, remove all of it
      if( operand2->isArray() )
      {
         CoreArray *removed = operand2->asArray();
         for( uint32 i = 0; i < removed->length(); i ++ )
         {
            dest->remove( removed->at(i) );
         }
      }
      else {
         dest->remove( *operand2 );
      }

      // never raise.
      vm->regA() = dest;
      return;
   }

   vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "X,X" ) ) );
}


/*#
   @function mul
   @brief Multiplies two items along the rules of VM '*' operator.
   @param a First operand
   @param b Second operand
   @return The result of the multiplication.

   This function operates also on strings, arrays and dictionaries
   replicating the behavior of the "*" operator as it is performed
   by the VM.
*/
FALCON_FUNC  fe_mul( ::Falcon::VMachine *vm )
{
   Item *operand1 = vm->param(0);
   Item *operand2 = vm->param(1);
   if ( operand2 == 0 || operand2 == 0 )
   {
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "X,X" ) ) );
      return;
   }


   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         vm->regA().setInteger( operand1->asInteger() * operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         vm->regA().setNumeric( operand1->asInteger() * operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         vm->regA().setNumeric( operand1->asNumeric() * operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         vm->regA().setNumeric( operand1->asNumeric() * operand2->asNumeric() );
      return;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_INT:
      case FLC_ITEM_STRING<< 8 | FLC_ITEM_NUM:
      {
         int64 chr = operand2->forceInteger();
         if ( chr >= 0 && chr <= (int64) 0xFFFFFFFF )
         {
            GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
            gcs->append( (uint32) chr );
            vm->regA().setString( gcs );
            return;
         }
      }
      break;
   }

   vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "X,X" ) ) );
}

/*#
   @function div
   @brief Divides two numeric operands along the rules of VM '/' operator.
   @param a First operand
   @param b Second operand
   @return The result of the division.

*/
FALCON_FUNC  fe_div( ::Falcon::VMachine *vm )
{
   Item *operand1 = vm->param(0);
   Item *operand2 = vm->param(1);
   if ( operand2 == 0 || operand2 == 0 )
   {
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "N,N" ) ) );
      return;
   }

   switch( operand2->type() )
   {
      case FLC_ITEM_INT:
      {
         int64 val2 = operand2->asInteger();
         if ( val2 == 0 ) {
            vm->raiseModError( new MathError( ErrorParam( e_div_by_zero ).origin( e_orig_runtime ) ) );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               vm->regA().setNumeric( operand1->asInteger() / (numeric)val2 );
            return;
            case FLC_ITEM_NUM:
               vm->regA().setNumeric( operand1->asNumeric() / (numeric)val2 );
            return;
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         numeric val2 = operand2->asNumeric();
         if ( val2 == 0.0 ) {
            vm->raiseRTError( new MathError( ErrorParam( e_div_by_zero ).origin( e_orig_vm ) ) );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               vm->regA().setNumeric( operand1->asInteger() / (numeric)val2 );
            return;
            case FLC_ITEM_NUM:
               vm->regA().setNumeric( operand1->asNumeric() / (numeric)val2 );
            return;
         }
      }
      break;
   }

   vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "N,N" ) ) );
}


/*#
   @function mod
   @brief Performs a modulo division on numeric operands along the rules of VM '%' operator.
   @param a First operand
   @param b Second operand
   @return The result of the modulo.

*/
FALCON_FUNC  fe_mod( ::Falcon::VMachine *vm )
{
   Item *operand1 = vm->param(0);
   Item *operand2 = vm->param(1);
   if ( operand2 == 0 || operand2 == 0 )
   {
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "N,N" ) ) );
      return;
   }

   if ( operand1->type() == FLC_ITEM_INT && operand2->type() == FLC_ITEM_INT ) {
      if ( operand2->asInteger() == 0 )
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MOD").origin( e_orig_vm ) ) );
      else
         vm->regA().setInteger( operand1->asInteger() % operand2->asInteger() );
   }
   else if ( operand1->isOrdinal() && operand2->isOrdinal() )
   {
      if ( operand2->forceNumeric() == 0.0 )
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MOD").origin( e_orig_vm ) ) );
      else
         vm->regA().setNumeric( fmod( operand1->forceNumeric(), operand2->forceNumeric() ) );
   }
   else
      vm->raiseModError( new ParamError(
         ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( "N,N" ) ) );
}

}
}


/* end of funcext_ext.cpp */
