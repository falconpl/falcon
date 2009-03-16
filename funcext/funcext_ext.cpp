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
   @brief Passe-par-tout accessor function.
   @param item An array, string, or any accessible item or collection.
   @param access A number, a range or a string to access the item.
   @optparam value If given, will substitue the given item or range (new value).
   @raise AccessError in case the accessor is invalid for the given item type.

   This function emulates all the language-level accessors provided by Falcon.
   Subscript accessors ([]) accepting numbers, ranges and generic items (for
   dictionaries) and property accessors (.) accepting strings are fully 
   supported. When two parameters are passed, the function works in 
   access semantics, while when the @b value parameter is also given,
   the function will work as an accessor/subscript assignment. In example,
   to change a string the @b at function can be used as a range accessor:

   @code
      string = "hello"
      string[0:1] = "H"          //first letter up
      at( string, [1:], "ELLO" ) // ...all up
      > string
      > "First letter: ", at( string, 0 ) 
                          // ^^^ same as string[0]
   @endcode

   This function is also able to access and modify the bindings of
   arrays (i.e. like accessing the arrays using the "." operator),
   members of objects and instances and static methods of classes.
   Properties and bindings can be accessed by names as strings. In
   example:
   
   @code
      // making a binding
      // ... equivalent to "array.bind = ..."
      array = []
      at( array, "bind", "binding value" )
      > array.bind
      
      //... accessing a property
      at( CurrentTime(), "toRFC2822" )()
   @endcode

   Trying to access an item with an incompatible type of accessor
   (i.e. trying to access an object with a range, or a string with a string).

   @note At the moment, the @b at function doesn't support BOM methods.
*/

FALCON_FUNC  fe_at ( ::Falcon::VMachine *vm )
{
   Item *i_array = vm->param(0);
   Item *i_pos = vm->param(1);
   Item *i_val = vm->param(2);

   CoreObject *self = 0;
   CoreClass *sourceClass=0;
   uint32 pos;

   if ( i_array == 0 || i_pos == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            origin( e_orig_runtime ).
            extra("X, N|R|S, [X]") ) );
      return;
   }

   switch( i_array->type() )
   {

   case FLC_ITEM_ARRAY:
   {
      CoreArray *ca = i_array->asArray();

      if( i_pos->isString() )
      {
         Item *found;
         if( ca->bindings() != 0 &&
            ( found = ca->bindings()->find( i_pos->asString() ) ) != 0 )
         {
            vm->regA() = *found->dereference();
            // propagate owner bindings
            if ( vm->regA().isArray() )
            {
               vm->regA().asArray()->setBindings( ca->bindings() );
            }
            if ( i_val != 0 )
               *found = *i_val;
            return;
         }
         else {
            if ( i_val != 0 ) {
               if ( ca->bindings() == 0 )
                  ca->setBindings( new LinearDict() );
               ca->bindings()->insert( i_pos->asString(), *i_val );
               vm->retnil();
               return;
            }
            else {
               vm->raiseModError( new AccessError( ErrorParam( e_prop_acc, __LINE__ ).
                  origin( e_orig_runtime ).
                  extra( *i_pos->asString() ) )
               );
               return;
            }
         }
      }
      else if ( i_pos->isOrdinal() )
      {
         int32 pos = (int32) i_pos->forceInteger();
         if ( pos < 0 ) pos = ca->length() - pos;
         if ( pos >= (int32) ca->length() )
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
         int64 start = i_pos->asRangeStart();
         int64 end = i_pos->asRangeIsOpen() ? ca->length() : i_pos->asRangeEnd();
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
         int32 pos = (int32) i_pos->forceInteger();
         if ( pos < 0 ) pos = str->length() - pos;
         if ( pos >= (int32) str->length() )
         {
            vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
               origin( e_orig_runtime ) )
               );
            return;
         }
         vm->retval( new CoreString( str->subString( pos, pos+1) ) );

         if ( i_val != 0 && i_val->isString() && i_val->asString()->size() > 0 )
         {
            str->setCharAt(pos, i_val->asString()->getCharAt(0) );
         }

         return;
      }
      else if ( i_pos->isRange() )
      {
         int64 start = i_pos->asRangeStart();
         int64 end = i_pos->asRangeIsOpen() ? str->length() : i_pos->asRangeEnd();

         vm->retval( new CoreString( str->subString( start, end ) ) );
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
      {
         CoreDict *dict = i_array->asDict();
         if( i_val == 0 )
         {
            if ( dict->find( *i_pos, vm->regA() ) )
               return;
         }
         else {
            bool find = dict->find( *i_pos, vm->regA() );
            dict->insert( *i_pos, *i_val );
            // assign the value itself if find is not found
            if (! find) 
               vm->retnil();
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
         CoreObject *self = i_array->asObject();
         Item prop;
         if( self->getProperty( *i_pos->asString(), prop ) )
         {
            // we must create a method if the property is a function.
            Item *p = prop.dereference();

            switch( p->type() ) {
               case FLC_ITEM_FUNC:
                  // the function may be a dead function; by so, the method will become a dead method,
                  // and it's ok for us.
                  vm->regA().setMethod( self, p->asFunction() );
                  break;

               case FLC_ITEM_CLASS:
                  vm->regA().setClassMethod( self, p->asClass() );
                  break;

               default:
                  vm->regA() = *p;
            }
            // we can set the property now
            if ( i_val != 0 )
            {
               if ( ! self->setProperty( *i_pos->asString(), *i_val ) )
               {
                  vm->raiseModError( new AccessError( ErrorParam( e_prop_ro, __LINE__ ).
                     origin( e_orig_runtime ).
                     extra( *i_pos->asString()) )
                     );
               }
            }
            //it's ok anyhow
            return;
         }

         vm->raiseModError( new AccessError( ErrorParam( e_prop_acc, __LINE__ ).
            origin( e_orig_runtime ).
            extra( *i_pos->asString()) )
            );
         return;
      }
      break;

   case FLC_ITEM_CLSMETHOD:
         sourceClass = i_array->asMethodClass();
         self = i_array->asObject();

   // do not break: fallback
   case FLC_ITEM_CLASS:
      if ( i_val != 0 )
      {
         vm->raiseModError( new AccessError( ErrorParam( e_prop_ro, __LINE__ ).
            origin( e_orig_runtime ).
            extra( *i_pos->asString()) )
            );
         return;
      }

      if ( sourceClass == 0 )
         sourceClass = i_array->asClass();

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
                  vm->regA().setMethod( self, prop->asFunction() );
               else
                  vm->regA().setFunction( prop->asFunction());
            }
            else
            {
               vm->regA() = *prop;
            }
            return;
         }

         vm->raiseModError( new AccessError( ErrorParam( e_prop_acc, __LINE__ ).
            origin( e_orig_runtime ).
            extra( *i_pos->asString() ))
            );
         return;
      }
      break;
   }

   vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ ).
      origin( e_orig_runtime ).
      extra("X, N|S|R, [X]") ) );
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

   vm->regA().setBoolean( i_a->compare( *i_b ) > 0 );
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

   vm->regA().setBoolean( i_a->compare( *i_b ) >= 0 );
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

   vm->regA().setBoolean( i_a->compare( *i_b ) < 0 );
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

   vm->regA().setBoolean( i_a->compare( *i_b ) <= 0 );
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

   vm->regA().setBoolean( i_a->compare( *i_b ) == 0 );
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

   vm->regA().setBoolean( i_a->compare( *i_b ) != 0 );
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
   if( first.compare( second ) == 0 )
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
            CoreString *gcs = new CoreString( *operand1->asString() );
            gcs->append( (uint32) chr );
            vm->regA().setString( gcs );
            return;
         }
      }
      break;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_STRING:
      {
         CoreString *gcs = new CoreString( *operand1->asString() );
         gcs->append( *operand2->asString() );
         vm->regA().setString( gcs );
      }
      return;

      case FLC_ITEM_DICT<< 8 | FLC_ITEM_DICT:
      {
         CoreDict *dict = new LinearDict( operand1->asDict()->length() + operand2->asDict()->length() );
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
            CoreString *gcs = new CoreString( *operand1->asString() );
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
