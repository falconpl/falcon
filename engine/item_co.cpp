/*
   FALCON - The Falcon Programming Language.
   FILE: item_co.cpp

   Item Common Operations support.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 02 Jan 2009 21:03:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/item.h>
#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/common.h>
#include <falcon/symbol.h>
#include <falcon/coreobject.h>
#include <falcon/carray.h>
#include <falcon/coredict.h>
#include <falcon/cclass.h>
#include <falcon/corefunc.h>
#include <falcon/membuf.h>
#include <falcon/error.h>
#include <falcon/lineardict.h>
#include <falcon/vm.h>

#include <errno.h>
#include <math.h>

#include <stdio.h>

namespace Falcon {

// Generic fail
void co_fail()
{
   throw new TypeError( ErrorParam( e_invop ) );
}

void co_fail_unb()
{
   throw new TypeError( ErrorParam( e_invop_unb ) );
}


//=============================================================
// Add
//

void co_int_add( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         third = first.asInteger() + second.asInteger();
         return;

      case FLC_ITEM_NUM:
         third = first.asInteger() + second.asNumeric();
         return;

      case FLC_ITEM_REFERENCE:
         co_int_add( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "ADD" ) );
}

void co_num_add( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         third = first.asNumeric() + second.asInteger();
         return;

      case FLC_ITEM_NUM:
         third = first.asNumeric() + second.asNumeric();
         return;

      case FLC_ITEM_REFERENCE:
         co_num_add( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "ADD" ) );
}


void co_string_add( const Item& first, const Item& second, Item& third )
{
   String *sf = first.asString();
   CoreString *dest = new CoreString( *sf );

   // Clones also garbage status

   const Item *op2 = second.dereference();

   if( op2->isString() )
   {
      dest->append( *op2->asString() );
   }
   else
   {
      String tgt;
      VMachine *vm = VMachine::getCurrent();
      if ( vm != 0 )
         vm->itemToString( tgt, op2 ); // may raise
      else
         op2->toString( tgt );

      dest->append( tgt );
   }

   third = dest;
}


void co_dict_add( const Item& first, const Item& second, Item& third )
{
   CoreDict *source = first.asDict();
   const Item *op2 = second.dereference();
   Item mth;

   if ( source->getMethod( OVERRIDE_OP_ADD, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   if ( op2->isDict() )
   {
      // adjust for self-operations (do not clone if target == source)
      if( third.isDict() && third.asDict() == source )
      {
         source->merge( *op2->asDict() );
      }
      else {
         LinearDict *dict = new LinearDict( source->length() + op2->asDict()->length() );
         dict->merge( source->items() );
         dict->merge( op2->asDict()->items() );
         third = new CoreDict( dict );
      }
      return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "ADD" ) );
}


void co_array_add( const Item& first, const Item& second, Item& third )
{
   // adjust for self-operations (do not clone if target == source)
   CoreArray *array = third.isArray() && third.asArray() == first.asArray() ?
         first.asArray() :
         first.asArray()->clone();
   const Item *op2 = second.dereference();

   if ( op2->isArray() )
   {
      array->merge( *op2->asArray() );
   }
   else
   {
      if ( op2->isString() )
         array->append( new CoreString( *op2->asString() ) );
      else
         array->append( *op2 );
   }

   third = array;
}

void co_object_add( const Item& first, const Item& second, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_ADD, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_ADD ) );
}

void co_ref_add( const Item& first, const Item& second, Item& third )
{
   Item& ref = first.asReference()->origin();
   ref.add( second, third );
}

void co_lbind_add( const Item& first, const Item& second, Item& third )
{
   Item& ref = first.asReference()->origin();
   ref.add( second, third );
}

void co_method_add( const Item& first, const Item& second, Item& third )
{
   if ( ! first.asMethodFunc()->isFunc() )
   {
      co_array_add( dyncast<CoreArray*>(first.asMethodFunc()), second, third );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra( "ADD" ) );
}

//=============================================================
// Sub
//

void co_int_sub( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         third.setInteger( first.asInteger() - second.asInteger() );
         return;

      case FLC_ITEM_NUM:
         third.setNumeric( first.asInteger() - second.asNumeric() );
         return;

      case FLC_ITEM_REFERENCE:
         co_int_sub( first, second.asReference()->origin(), third );
         return;
    }

    throw new TypeError( ErrorParam( e_invop ).extra("SUB") );
}

void co_num_sub( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         third.setNumeric( first.asNumeric() - second.asInteger() );
         return;

      case FLC_ITEM_NUM:
         third.setNumeric( first.asNumeric() - second.asNumeric() );
         return;

      case FLC_ITEM_REFERENCE:
         co_num_sub( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra("SUB") );
}

void co_array_sub( const Item& first, const Item& second, Item& third )
{
   // remove any element from an array
   CoreArray *source = first.asArray();
   // accomodate for selfsub
   CoreArray *dest = third.isArray() && third.asArray() == source ?
      source :
      source->clone();
   const Item *op2 = second.dereference();

   // if we have an array, remove all of it
   if( op2->isArray() )
   {
      CoreArray *removed = op2->asArray();
      for( uint32 i = 0; i < removed->length(); i ++ )
      {
         int32 rem = dest->find( removed->at(i) );
         if( rem >= 0 )
            dest->remove( rem );
      }
   }
   else
   {
      int32 rem = dest->find( *op2 );
      if( rem >= 0 )
         dest->remove( rem );
   }

   third = dest;
}


void co_dict_sub( const Item& first, const Item& second, Item& third )
{
   // remove various keys from arrays
   CoreDict *source = first.asDict();
   // accomodate for selfsub
   CoreDict *dest = third.isDict() && third.asDict() == source ? source : source->clone();
   const Item *op2 = second.dereference();
   Item mth;

   if ( source->getMethod( OVERRIDE_OP_SUB, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   // if we have an array, remove all of it
   if( op2->isArray() )
   {
      CoreArray *removed = op2->asArray();
      for( uint32 i = 0; i < removed->length(); i ++ )
      {
         dest->remove( removed->at(i) );
      }
   }
   else {
      dest->remove( *op2 );
   }

   // never raise.
   third = dest;
}


void co_object_sub( const Item& first, const Item& second, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_SUB, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_SUB ) );
}

void co_ref_sub( const Item& first, const Item& second, Item& third )
{
   Item& ref = first.asReference()->origin();
   ref.sub( second, third );
}

void co_method_sub( const Item& first, const Item& second, Item& third )
{
   if ( ! first.asMethodFunc()->isFunc() )
   {
      co_array_sub( dyncast<CoreArray*>(first.asMethodFunc()), second, third );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra( "ADD" ) );
}

//=============================================================
// MUL
//

void co_int_mul( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         third = first.asInteger() * second.asInteger();
         return;

      case FLC_ITEM_NUM:
         third = first.asInteger() * second.asNumeric();
         return;

      case FLC_ITEM_REFERENCE:
         co_int_mul( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "MUL" ) );
}

void co_num_mul( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         third = first.asNumeric() * second.asInteger();
         return;

      case FLC_ITEM_NUM:
         third = first.asNumeric() * second.asNumeric();
         return;

      case FLC_ITEM_REFERENCE:
         co_num_mul( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "MUL" ) );
}

void co_string_mul( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
      case FLC_ITEM_NUM:
         {
            String *str = first.asString();
            int64 chr = second.forceInteger();
            if ( chr >= 0 && chr <= (int64) 0xFFFFFFFF )
            {
               CoreString *gcs = new CoreString( str->length() );
               for ( int i = 0; i < chr; i ++ )
               {
                  gcs->append( *str );
               }
               third = gcs;
               return;
            }
            break;
         }

      case FLC_ITEM_REFERENCE:
         co_string_mul( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "MUL" ) );
}


void co_dict_mul( const Item& first, const Item& second, Item& third )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_MUL, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_MUL ) );
}


void co_object_mul( const Item& first, const Item& second, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_MUL, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_MUL ) );
}


void co_ref_mul( const Item& first, const Item& second, Item& third )
{
   Item& ref = first.asReference()->origin();
   ref.mul( second, third );
}

//=============================================================
// DIV
//

void co_int_div( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
      {
         int64 val2 = second.asInteger();
         if ( val2 == 0 ) {
            throw new MathError( ErrorParam( e_div_by_zero ) );
         }
         third.setNumeric( first.asInteger() / (numeric)val2 );

         return;
      }

      case FLC_ITEM_NUM:
      {
         numeric val2 = second.asNumeric();
         if ( val2 == 0.0 ) {
            throw new MathError( ErrorParam( e_div_by_zero ) );
            return;
         }
         third.setNumeric( first.asInteger() / val2 );

         return;
      }

      case FLC_ITEM_REFERENCE:
         co_int_div( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "DIV" ) );
}


void co_num_div( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
      {
         int64 val2 = second.asInteger();
         if ( val2 == 0 ) {
            throw new MathError( ErrorParam( e_div_by_zero ) );
         }
         third.setNumeric( first.asNumeric() / (numeric)val2 );

         return;
      }

      case FLC_ITEM_NUM:
      {
         numeric val2 = second.asNumeric();
         if ( val2 == 0.0 ) {
            throw new MathError( ErrorParam( e_div_by_zero ) );
         }
         third.setNumeric( first.asNumeric() / val2 );

         return;
      }

      case FLC_ITEM_REFERENCE:
         co_num_div( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "DIV" ) );
}


void co_string_div( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
      case FLC_ITEM_NUM:
         {
            String *str = first.asString();
            int64 chr = second.forceInteger();
            uint32 len = str->length();
            if ( chr >= -(int64) 0xFFFFFFFF && chr <= (int64) 0xFFFFFFFF && len > 0 )
            {
               CoreString *gcs = new CoreString( *str );
               gcs->setCharAt( len-1, gcs->getCharAt(len-1) + (uint32)chr );
               third = gcs;
               return;
            }
            break;
         }

      case FLC_ITEM_REFERENCE:
         co_string_div( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "DIV" ) );
}

void co_dict_div( const Item& first, const Item& second, Item& third )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_DIV, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_DIV ) );
}


void co_object_div( const Item& first, const Item& second, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_DIV, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_DIV ) );
}

void co_ref_div( const Item& first, const Item& second, Item& third )
{
   Item& ref = first.asReference()->origin();
   ref.div( second, third );
}

//=============================================================
// MOD
//

void co_int_mod( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         if ( second.asInteger() == 0 )
            throw new MathError( ErrorParam( e_div_by_zero ).extra("MOD") );

         third.setInteger( first.asInteger() % second.asInteger() );
         return;

      case FLC_ITEM_NUM:
         if ( second.asNumeric() == 0.0 )
            throw new MathError( ErrorParam( e_div_by_zero ).extra("MOD") );

         third.setInteger( first.asInteger() % (int64) second.asNumeric() );
         return;

      case FLC_ITEM_REFERENCE:
         co_int_mod( first, second.asReference()->origin(), third );
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "MOD" ) );
}


void co_num_mod( const Item& first, const Item& second, Item& third )
{
   switch( second.type() )
   {
      case FLC_ITEM_INT:
         if ( second.asInteger() == 0 )
            throw new MathError( ErrorParam( e_div_by_zero ).extra("MOD") );

         third.setInteger( ((int64)first.asNumeric()) % second.asInteger() );
         return;

      case FLC_ITEM_NUM:
         if ( second.asNumeric() == 0.0 )
            throw new MathError( ErrorParam( e_div_by_zero ).extra("MOD") );

         third.setInteger( ((int64)first.asNumeric()) % (int64) second.asNumeric() );
         return;

      case FLC_ITEM_REFERENCE:
         co_num_mod( first, second.asReference()->origin(), third );
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "MOD" ) );
}


void co_string_mod( const Item& first, const Item& second, Item& third )
{
   String *str = first.asString();

   switch( second.type() )
   {
      case FLC_ITEM_INT:
         {
            int64 chr = second.asInteger();
            if ( chr >= 0 && chr <= (int64) 0xFFFFFFFF )
            {
               CoreString *gcs = new CoreString( *str );
               gcs->append( (uint32) chr );
               third = gcs;
               return;
            }
            break;
         }

      case FLC_ITEM_NUM:
         {
            numeric chr = second.asNumeric();
            if ( chr >= 0 && chr <= (numeric) 0xFFFFFFFF )
            {
               CoreString *gcs = new CoreString( *str );
               gcs->append( (uint32) chr );
               third = gcs;
               return;
            }
            break;
         }

      case FLC_ITEM_REFERENCE:
         co_string_mod( first, second.asReference()->origin(), third );
         return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra( "MOD" ) );
}


void co_dict_mod( const Item& first, const Item& second, Item& third )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_MOD, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_MOD ) );
}


void co_object_mod( const Item& first, const Item& second, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_MOD, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_MOD ) );
}

void co_ref_mod( const Item& first, const Item& second, Item& third )
{
   Item& ref = first.asReference()->origin();
   ref.mod( second, third );
}

//=============================================================
// pow
//

void co_int_pow( const Item& first, const Item& second, Item& third )
{
   numeric powval;

   switch( second.type() )
   {
      case FLC_ITEM_INT:
         if ( second.asInteger() == 0 ) {
            third = (int64) 1;
            return;
         }
         powval = pow( (double)first.asInteger(), (numeric) second.asInteger() );
         break;

      case FLC_ITEM_NUM:
         if ( second.asNumeric() == 0.0 ) {
            third = (int64) 1;
            return;
         }
         powval = pow( (double)first.asInteger(), (numeric) second.asNumeric() );
         break;

      case FLC_ITEM_REFERENCE:
         co_int_pow( first, second.asReference()->origin(), third );
         return;


      default:
         throw new TypeError( ErrorParam( e_invop ).extra( "POW" ) );
   }

   if ( errno != 0 )
   {
      errno = 0;
      throw new MathError( ErrorParam( e_domain ).extra( "POW" ) );
   }

   third = powval;
}


void co_num_pow( const Item& first, const Item& second, Item& third )
{
   numeric powval;

   switch( second.type() )
   {
      case FLC_ITEM_INT:
         if ( second.asInteger() == 0 ) {
            third = (int64) 1;
            return;
         }
         powval = pow( first.asNumeric(), (numeric) second.asInteger() );
         break;

      case FLC_ITEM_NUM:
         if ( second.asNumeric() == 0.0 ) {
            third = (int64) 1;
            return;
         }
         powval = pow( first.asNumeric(), (numeric) second.asNumeric() );
         break;

      case FLC_ITEM_REFERENCE:
         co_num_pow( first, second.asReference()->origin(), third );
         return;


      default:
         throw new TypeError( ErrorParam( e_invop ).extra( "POW" ) );
   }

   if ( errno != 0 )
   {
      errno = 0;
      throw new MathError( ErrorParam( e_domain ).extra( "POW" ) );
   }

   third = powval;
}


void co_dict_pow( const Item& first, const Item& second, Item& third )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_POW, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_POW ) );
}


void co_object_pow( const Item& first, const Item& second, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_POW, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *second.dereference() );
         vm->callItemAtomic( mth, 1 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_POW ) );
}

void co_ref_pow( const Item& first, const Item& second, Item& third )
{
   Item& ref = first.asReference()->origin();
   ref.pow( second, third );
}


//=============================================================
// NEG
//

void co_int_neg( const Item& first, Item& tgt )
{
   tgt = -first.asInteger();
}

void co_num_neg( const Item& first, Item& tgt )
{
   tgt = -first.asNumeric();
}


void co_dict_neg( const Item& first, Item& tgt )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_NEG, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         tgt = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_NEG ) );
}


void co_object_neg( const Item& first, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_NEG, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_NEG ) );
}

void co_ref_neg( const Item& first, Item &tgt )
{
   Item& ref = first.asReference()->origin();
   ref.neg( tgt );
}


//=============================================================
// INC (prefix)
//

void co_int_inc( Item& first, Item &target )
{
   first = first.asInteger() + 1;
   target = first;
}

void co_num_inc( Item& first, Item &target )
{
   first = first.asNumeric() + 1.0;
   target = first;
}


void co_dict_inc( Item& first, Item &target )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_INC, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         target = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_INC ) );
}


void co_object_inc( Item& first, Item &target )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_INC, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         target = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_INC ) );
}

void co_ref_inc( Item& first, Item &target )
{
   Item& ref = first.asReference()->origin();
   ref.inc(target);
}


//=============================================================
// DEC (prefix)
//

void co_int_dec( Item& first, Item &target )
{
   first = first.asInteger() - 1;
   target = first;
}

void co_num_dec( Item& first, Item &target )
{
   first = first.asNumeric() - 1.0;
   target = first;
}


void co_dict_dec( Item& first, Item &target )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_DEC, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         vm->callItemAtomic( mth, 0 );
         target = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_DEC ) );
}


void co_object_dec( Item& first, Item &target )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_DEC, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         target = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_DEC ) );
}

void co_ref_dec( Item& first, Item &target )
{
   Item& ref = first.asReference()->origin();
   ref.dec( target );
}


//=============================================================
// INC (postfix)
//

void co_int_incpost( Item& first, Item& tgt )
{
   tgt = first;
   first = first.asInteger() + 1;
}

void co_num_incpost( Item& first, Item& tgt )
{
   tgt = first;
   first = first.asNumeric() + 1.0;
}


void co_dict_incpost( Item& first, Item& third )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_INCPOST, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_INCPOST ) );
}



void co_object_incpost( Item& first, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_INCPOST, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_INCPOST ) );
}

void co_ref_incpost( Item& first, Item &tgt )
{
   Item& ref = first.asReference()->origin();
   ref.incpost( tgt );
}


//=============================================================
// DEC (postfix)
//

void co_int_decpost( Item& first, Item& tgt )
{
   tgt = first;
   first = first.asInteger() - 1;
}

void co_num_decpost( Item& first, Item& tgt )
{
   tgt = first;
   first = first.asNumeric() - 1.0;
}


void co_dict_decpost( Item& first, Item& third )
{
   CoreDict *self = first.asDict();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_DECPOST, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_DECPOST ) );
}


void co_object_decpost( Item& first, Item& third )
{
   CoreObject *self = first.asObjectSafe();

   Item mth;
   if ( self->getMethod( OVERRIDE_OP_DECPOST, mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->callItemAtomic( mth, 0 );
         third = vm->regA();
         return;
      }
   }

   throw new TypeError( ErrorParam( e_invop ).extra( OVERRIDE_OP_DECPOST ) );
}

void co_ref_decpost( Item& first, Item &tgt )
{
   Item& ref = first.asReference()->origin();
   ref.decpost( tgt );
}


//=============================================================
// Compare
//

int co_nil_compare( const Item& first, const Item& second )
{
   switch( second.type() )
   {
   case FLC_ITEM_REFERENCE:
      return co_nil_compare( first, second.asReference()->origin() );

   case FLC_ITEM_UNB:
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }

   return first.type() - second.type();
}

int co_bool_compare( const Item& first, const Item& second )
{
   // true > false

   switch( second.type() )
   {
      case FLC_ITEM_NIL:
         return 1;

      case FLC_ITEM_BOOL:
         return (first.asBoolean() ? 1 : 0) -
            (second.asBoolean() ? 1: 0);

      case FLC_ITEM_REFERENCE:
         return co_bool_compare( first, second.asReference()->origin() );

      case FLC_ITEM_UNB:
         const_cast<Item*>(&second)->copy( first );
         return 0;
   }

   // bool is always lower
   return -1;
}

int co_int_compare( const Item& first, const Item& second )
{
   switch ( second.type() )
   {
      case FLC_ITEM_NIL:
      case FLC_ITEM_BOOL:
         return 1;

      case FLC_ITEM_INT:
      {
         // converting the int to a 32 bit integer is not a simple matter
         int64 v = (first.asInteger() - second.asInteger()) ;
         if (v==0) return 0;
         if ( v > 0 ) return 1;
         return -1;
      }

      case FLC_ITEM_NUM:
         return (int)(((numeric)first.asInteger()) - second.asNumeric());

      case FLC_ITEM_REFERENCE:
         return co_int_compare( first, second.asReference()->origin() );

      case FLC_ITEM_UNB:
         const_cast<Item*>(&second)->copy( first );
         return 0;
      }

   // lower than the others
   return -1;
}

int co_num_compare( const Item& first, const Item& second )
{
   switch ( second.type() )
   {
      case FLC_ITEM_NIL:
      case FLC_ITEM_BOOL:
         return 1;

      case FLC_ITEM_INT:
         if( first.asNumeric() < ((numeric)second.asInteger()) )
            return -1;
         if( first.asNumeric() > ((numeric)second.asInteger()) )
            return 1;
         return 0;

      case FLC_ITEM_NUM:
         if (first.asNumeric() < second.asNumeric())
            return -1;
         if( first.asNumeric() > second.asNumeric())
            return 1;
         return 0;


      case FLC_ITEM_REFERENCE:
         return co_num_compare( first, second.asReference()->origin() );

      case FLC_ITEM_UNB:
         const_cast<Item*>(&second)->copy( first );
         return 0;
    }

   // lower than the others
   return -1;
}


int co_range_compare( const Item& first, const Item& second )
{
   switch ( second.type() )
   {
      case FLC_ITEM_NIL:
      case FLC_ITEM_BOOL:
      case FLC_ITEM_INT:
      case FLC_ITEM_NUM:
         return 1;

      case FLC_ITEM_RANGE:
      {
         int64 diff = first.asRangeStart() - second.asRangeStart();
         if ( diff != 0 )
            return (int)(diff>>32);

         // always greater
         diff =  (first.asRangeIsOpen() ? 1: 0) - (second.asRangeIsOpen() ? 1: 0);

         if ( diff != 0 )
            return (int)(diff);

         if ( first.asRangeIsOpen() )
            return 0;

         diff = first.asRangeEnd() - second.asRangeEnd();
         if ( diff != 0 )
            return (int)(diff>>32);

         return (int)((first.asRangeStep() - second.asRangeStep())>>32);
      }
      break;

      case FLC_ITEM_REFERENCE:
         return co_range_compare( first, second.asReference()->origin() );

      case FLC_ITEM_UNB:
         const_cast<Item*>(&second)->copy( first );
         return 0;
   }

   return -1;
}

int co_lbind_compare( const Item& first, const Item& second )
{
   switch( second.type() )
   {

   case FLC_ITEM_REFERENCE:
      return co_lbind_compare( first, second.asReference()->origin() );

   case FLC_ITEM_LBIND:
      return (int)(first.asLBind() - second.asLBind());

   case FLC_ITEM_UNB:
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }

   return first.type() - second.type();
}

int co_func_compare( const Item& first, const Item& second )
{
   switch( second.type() )
   {

   case FLC_ITEM_REFERENCE:
      return co_func_compare( first, second.asReference()->origin() );

   case FLC_ITEM_FUNC:
      return first.asFunction()->name().compare(second.asFunction()->name());

   case FLC_ITEM_UNB:
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }

   return first.type() - second.type();
}

int co_gcptr_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_gcptr_compare( first, second.asReference()->origin() );
   }
   else if ( second.isGCPointer() )
   {
      return (int)(first.asGCPointer() - second.asGCPointer());
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }

   return first.type() - second.type();
}


int co_string_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_string_compare( first, second.asReference()->origin() );
   }
   else if ( second.isString() )
   {
      return first.asString()->compare( *second.asString() );
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }

   return first.type() - second.type();
}


int co_array_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_array_compare( first, second.asReference()->origin() );
   }
   else if ( second.isArray() )
   {
      return first.asArray()->compare( *second.asArray() );
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }

   return (first.type() - second.type());
}


int co_dict_compare( const Item& first, const Item& second )
{
   CoreDict *self = first.asDict();
   const Item* psecond = second.dereference();

   if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }

   Item mth;
   if ( self->getMethod( "compare", mth ) )
   {
      VMachine *vm = VMachine::getCurrent();
      if ( vm !=  0 )
      {
         vm->pushParam( *psecond );
         vm->callItemAtomic( mth, 1 );
         if ( ! vm->regA().isNil() )
            return (int) vm->regA().forceInteger();

      }
   }

   if ( psecond->isDict() )
   {
      return first.asDict()->compare(psecond->asDict());
   }

   return first.type() - psecond->type();
}


int co_object_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_object_compare( first, second.asReference()->origin() );
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }
   else
   {
      CoreObject *self = first.asObjectSafe();

      // we're possibly calling a deep method,
      // and second may come from the stack.
      Item temp = second;

      // do we have an active VM?
      VMachine *vm = VMachine::getCurrent();
      if ( vm != 0 )
      {
         // first provides a less operator?
         Item mth;
         if ( self->getMethod( "compare", mth ) )
         {
            vm->pushParam( temp );
            vm->callItemAtomic( mth, 1 );
            if ( vm->regA().isInteger() )
            {
               return (int)vm->regA().asInteger();
            }
         }
      }

      // by fallback -- use normal ordering.
      if( temp.isObject() )
         return (int)(self - temp.asObjectSafe());
   }

   return first.type() - second.type();
}


int co_membuf_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_membuf_compare( first, second.asReference()->origin() );
   }
   else if ( second.isMemBuf() )
   {
      return (int)(first.asMemBuf() - second.asMemBuf());
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }


   return (int)(first.type() - second.type());
}

int co_ref_compare( const Item& first, const Item& second )
{
   const Item &ref = first.asReference()->origin();
   return ref.compare( second );
}

int co_clsmethod_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_clsmethod_compare( first, second.asReference()->origin() );
   }
   else if ( second.isClassMethod() )
   {
      return (int)(first.asMethodFunc() - second.asMethodFunc());
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }


   return first.type() - second.type();
}

int co_method_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_method_compare( first, second.asReference()->origin() );
   }
   else if ( second.isMethod() )
   {
      int cp =  first.asMethodItem().compare(second.asMethodItem());
      if ( cp == 0 )
         return (int)(first.asMethodFunc() - second.asMethodFunc());
      return cp;
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }


   return first.type() - second.type();
}

int co_class_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
   {
      return co_class_compare( first, second.asReference()->origin() );
   }
   else if ( second.isClass() )
   {
      return (int)(first.asClass()->symbol()->name().compare(second.asClass()->symbol()->name()));
   }
   else if ( second.isUnbound() )
   {
      const_cast<Item*>(&second)->copy( first );
      return 0;
   }


   return (int)(first.type() - second.type());
}


int co_unb_compare( const Item& first, const Item& second )
{
   if ( second.isReference() )
      return co_unb_compare( first, second.asReference()->origin() );

   const_cast<Item*>(&first)->copy( second );
   return 0;
}

//=============================================================
// Get Index
//

void co_range_getindex( const Item &item, const Item &idx, Item &result )
{
   int32 pos;
   switch( idx.type() )
   {
      case FLC_ITEM_INT:
         pos = (int32) idx.asInteger();
         break;

      case FLC_ITEM_NUM:
         pos = (int32) idx.asNumeric();
         break;

      case FLC_ITEM_REFERENCE:
         co_range_getindex( item, idx.asReference()->origin(), result );
         return;

      default:
         throw new AccessError( ErrorParam( e_arracc ).extra( "LDV" ) );
   }

   switch( pos )
   {
      case -3: case 0: result = (int64) item.asRangeStart(); return;

      case 1: case -2:
         if( item.asRangeIsOpen() )
            result.setNil();
         else
            result = (int64) item.asRangeEnd();
      return;

      case 2: case -1:
         if( item.asRangeIsOpen() )
            result.setNil();
         else
            result = (int64) item.asRangeStep();
      return;
   }

   throw new AccessError( ErrorParam( e_arracc ).extra( "LDV" ) );
}

void co_string_getindex( const Item &item, const Item &idx, Item &result )
{
   item.asString()->readIndex( idx, result );
}

void co_deep_getindex( const Item &item, const Item &idx, Item &result )
{
   item.asDeepItem()->readIndex( idx, result );
}

void co_method_getindex( const Item &item, const Item &idx, Item &result )
{
   if ( ! item.asMethodFunc()->isFunc() )
   {
      // an array
      dyncast<CoreArray*>(item.asMethodFunc())->readIndex( idx, result );
   }
   else
      throw new AccessError( ErrorParam( e_arracc ).extra( "LDV" ) );
}

void co_ref_getindex( const Item &item, const Item &idx, Item &result )
{
   item.asReference()->origin().getIndex( idx, result );
}


//=============================================================
// Set Index
//

void co_range_setindex( Item &item, const Item &idx, const Item &nval )
{
   int32 pos;
   switch( idx.type() )
   {
      case FLC_ITEM_INT:
         pos = (int32) idx.asInteger();
         break;

      case FLC_ITEM_NUM:
         pos = (int32) idx.asNumeric();
         break;

      case FLC_ITEM_REFERENCE:
         co_range_setindex( item, idx.asReference()->origin(), nval );
         return;

      default:
         throw new AccessError( ErrorParam( e_arracc ).extra( "STV" ) );
   }
   
   int32 value;
   switch( nval.type() )
   {
      case FLC_ITEM_NIL:
         if ( pos == -1 || pos == 2 )
         item.asRange()->setOpen();
         // no more things to do, let's return.
         return;
         
      case FLC_ITEM_INT:
         value = (int32) nval.asInteger();
         break;

      case FLC_ITEM_NUM:
         value = (int32) nval.asNumeric();
         break;

      case FLC_ITEM_REFERENCE:
         co_range_setindex( item, idx, nval.asReference()->origin() );
         return;

      default:
         throw new TypeError( ErrorParam( e_param_type ).extra( "STV" ) );
   }

   CoreRange *cr = item.asRange();

   switch( pos )
   {
      case -3: case 0:
         item.setRange( new CoreRange( value, cr->end(), cr->step() ) );
         return;

      case 1: case -2:
         item.setRange( new CoreRange( cr->start(), value, cr->step() ) );
         return;

      case 2: case -1:
         item.setRange( new CoreRange( cr->start(), cr->end(), value ) );
         return;
   }

   throw new AccessError( ErrorParam( e_arracc ).extra( "STV" ) );
}

void co_string_setindex( const Item &item, const Item &idx, Item &result )
{
   // alter the item so that it is now a core string.
   if( ! item.asString()->isCore() )
   {
      const_cast<Item *>(&item )->setString( new CoreString( *item.asString() ) );
   }

   item.asString()->writeIndex( idx, result );
}


void co_deep_setindex( Item &item, const Item &idx, Item &result )
{
   item.asDeepItem()->writeIndex( idx, result );
}

void co_method_setindex( const Item &item, const Item &idx, Item &result )
{
   if ( ! item.asMethodFunc()->isFunc() )
   {
      // an array
      dyncast<CoreArray*>(item.asMethodFunc())->writeIndex( idx, result );
   }
   else
      throw new AccessError( ErrorParam( e_arracc ).extra( "STV" ) );
}

void co_ref_setindex( Item &item, const Item &idx, Item &result )
{
   item.asReference()->origin().setIndex( idx, result );
}

//=============================================================
// Get Property
//
void co_generic_getproperty( const Item &item, const String &prop, Item &result )
{
   VMachine* vm = VMachine::getCurrent();
   fassert( vm != 0 );
   CoreClass* cs = vm->getMetaClass( item.type() );
   if ( cs != 0 )
   {
      uint32 pos;
      if( cs->properties().findKey( prop, pos ) )
      {
         Item *prop = cs->properties().getValue( pos );
         fassert( prop->isFunction() );
         result.setMethod( item, prop->asFunction() );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
}

void co_string_getproperty( const Item &item, const String &prop, Item &result )
{
   item.asString()->readProperty( prop, result );
}


void co_deep_getproperty( const Item &item, const String &prop, Item &result )
{
   item.asDeepItem()->readProperty( prop, result );
}

void co_ref_getproperty( const Item &item, const String &idx, Item &result )
{
   item.asReference()->origin().getProperty( idx, result );
}

void co_method_getproperty( const Item &item, const String &prop, Item &result )
{
   VMachine* vm = VMachine::getCurrent();
   fassert( vm != 0 );
   CoreClass* cs = vm->getMetaClass( FLC_ITEM_METHOD );
   if ( cs != 0 )
   {
      uint32 pos;
      if( cs->properties().findKey( prop, pos ) )
      {
         Item *prop = cs->properties().getValue( pos );
         fassert( prop->isFunction() );
         result.setMethod( new GarbageItem( item ), prop->asFunction() );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( prop ) );
}

void co_classmeth_getproperty( const Item &item, const String &idx, Item &target )
{

   CoreClass* sourceClass = item.asMethodClass();
   CoreObject *self = item.asMethodClassOwner();
   uint32 pos;

   if( sourceClass->properties().findKey( idx, pos ) )
   {
      Item *prop = sourceClass->properties().getValue( pos );

      // now, accessing a method in a class means that we want to call the base method in a
      // self item:
      switch( prop->type() ) {
         case FLC_ITEM_FUNC:
            // the function may be a dead function; by so, the method will become a dead method,
            // and it's ok for us.
            target.setMethod( self, prop->asFunction() );
            break;

         case FLC_ITEM_CLASS:
            target.setClassMethod( self, prop->asClass() );
            break;

         default:
            target = *prop;
         }

      return;
   }

   // try to find a generic method
   VMachine* vm = VMachine::getCurrent();
   fassert( vm != 0 );
   CoreClass* cs = vm->getMetaClass( FLC_ITEM_CLSMETHOD );
   if ( cs != 0 )
   {
      uint32 pos;
      if( cs->properties().findKey( idx, pos ) )
      {
         Item *prop = cs->properties().getValue( pos );
         fassert( prop->isFunction() );
         target.setMethod( new GarbageItem( item ), prop->asFunction() );
         return;
      }
   }

   throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( idx ) );
}


void co_class_getproperty( const Item &item, const String &idx, Item &result )
{
   CoreClass *sourceClass = item.asClass();
   uint32 pos;

   if( sourceClass->properties().findKey( idx, pos ) )
   {
      Item *prop = sourceClass->properties().getValue( pos );

      switch( prop->type() )
      {
         // it is legal to get methods in classes; We can have static methods.
         case FLC_ITEM_FUNC:
            {
               result.setMethod( item, prop->asFunction() );
            }
            break;

         // getting a class in a class creates a classMethod, as getting a class in
         // an object.
         case FLC_ITEM_CLASS:
            {
               VMachine* vm = VMachine::getCurrent();
               fassert( vm != 0 );
               if ( vm->self().isObject() )
                  result.setClassMethod( vm->self().asObjectSafe(), prop->asClass() );
               else
                  result = *prop;
            }
            break;

         default:
            result = *prop;
         }

      return;
   }

   // try to find a generic method
   VMachine* vm = VMachine::getCurrent();
   fassert( vm != 0 );
   CoreClass* cc = vm ->getMetaClass( FLC_ITEM_CLASS );

   uint32 id;
   if ( cc != 0 && cc->properties().findKey( idx, id ) )
   {
      Item* p = cc->properties().getValue( id );
      fassert( ! p->isReference() );
      result.setMethod( sourceClass, p->asFunction() );
      return;
   }

   throw new AccessError( ErrorParam( e_prop_acc, __LINE__ ).extra( idx ) );
}

//=============================================================
// Write Property
//


void co_deep_setproperty( Item &item, const String &idx, const Item &result )
{
   item.asDeepItem()->writeProperty( idx, result );
}

void co_ref_setproperty( Item &item, const String &idx, const Item &result )
{
   item.asReference()->origin().setProperty( idx, result );
}

void co_class_setproperty( Item &item, const String &idx, const Item &result )
{
   // is the target property a static property -- it must be a reference.
   PropertyTable &pt = item.asClass()->properties();
   uint32 propId;
   if ( pt.findKey( idx, propId ) )
   {
      Item *prop = pt.getValue( propId );
      if ( prop->isReference() )
      {
         if ( result.isString() )
         {
           *prop->dereference() = new CoreString( *result.asString() );
         }
         else
            *prop->dereference() = *result.dereference();
         return;
      }
      else {
         throw
            new AccessError( ErrorParam( e_prop_ro ).origin( e_orig_vm )
                .extra( idx ) );
         return;
      }
   }

   throw
      new AccessError( ErrorParam( e_prop_acc ).origin( e_orig_vm ).
         extra( idx ) );
}

//=============================================================
// Call
//

void co_call_uncallable( VMachine *vm, uint32 paramCount )
{
   //TODO: Useful? -- on throw we either unroll or close the VM...
   /*
   if ( paramCount != 0 )
      vm->currentStack().resize( vm->currentStack().size() - paramCount );
   */

   // TODO: correct error.
   throw new TypeError( ErrorParam( e_invop ).extra("CALL") );
}

void co_call_function( const Item &itm, VMachine *vm, uint32 paramCount )
{
   // fill - in the missing parameters.
   vm->prepareFrame( itm.asFunction(), paramCount );
   // leave self as is.
}

void co_call_reference( const Item &itm, VMachine *vm, uint32 paramCount )
{
   itm.asReference()->origin().readyFrame( vm, paramCount );
}


void co_call_array( const Item &itm, VMachine *vm, uint32 paramCount )
{
   CoreArray *arr = itm.asArray();


   if ( arr->length() != 0 )
   {
      const Item &carr = arr->at(0);

      if ( carr.asArray() != arr && carr.isCallable() )
      {
         vm->prepareFrame( arr, paramCount );

         // the prepare for array already checks for self -- tables
         if ( arr->table() != 0 )
            vm->self() = arr->table();

         return;
      }
   }

   // TODO: correct error.
   throw new TypeError( ErrorParam( e_invop ).extra("CALL") );
}


void co_call_dict( const Item &itm, VMachine *vm, uint32 paramCount )
{
   // find the call__ member, if it exists.
   CoreDict *self = itm.asDict();

   Item *mth;
   if ( (mth = self->find( OVERRIDE_OP_CALL ) ) != 0 && mth->isCallable() )
   {
      mth->readyFrame( vm, paramCount );
      vm->self() = self;
      return;
   }

   // TODO: correct error.
   throw new TypeError( ErrorParam( e_invop ).extra("CALL") );
}


void co_call_object( const Item &itm, VMachine *vm, uint32 paramCount )
{
   // find the call__ member, if it exists.
   CoreObject *self = itm.asObjectSafe();

   Item mth;
   if ( self->getProperty( OVERRIDE_OP_CALL, mth ) && mth.isCallable() )
   {
      mth.readyFrame( vm, paramCount );
      vm->self() = self;
      return;
   }

   // TODO: correct error.
   throw new TypeError( ErrorParam( e_invop ).extra("CALL") );
}

void co_call_method( const Item &itm, VMachine *vm, uint32 paramCount )
{
   // fill - in the missing parameters.
   itm.asMethodFunc()->readyFrame( vm, paramCount );
   itm.getMethodItem( vm->self() );
}

static bool __call_init_enter_last( VMachine *vm )
{
   vm->regA() = vm->self();
   return false;
}

static bool __call_init_enter( VMachine *vm )
{
   Item initEnter;
   if( vm->self().asObject()->getMethod( "__enter", initEnter ) )
   {
      vm->returnHandler(0);
      vm->callFrame( initEnter, 0 );
      return true;
   }

   vm->regA() =  vm->self();
   return false;
}

void co_call_class( const Item &itm, VMachine *vm, uint32 paramCount )
{
   CoreClass *cls = itm.asClass();
   CoreObject* inst = cls->createInstance();
   fassert( inst != 0 );

   // if the class has not a constructor, we just set the item in A
   // and return
   if ( cls->constructor().isNil() )
   {
      // we are sure it's a core object
      vm->regA().setObject( inst );

      // pop the stack
      vm->stack().resize( vm->stack().length() - paramCount );
      if( cls->hasInitEnter() )
      {
         Item initEnter;
         if( inst->getProperty( "__enter", initEnter ) && initEnter.isFunction() )
         {
            vm->prepareFrame( initEnter.asFunction(), 0 );
            vm->self() = inst;
            inst->gcMark( memPool->generation() );
            vm->returnHandler( __call_init_enter_last );
         }
      }

      return;
   }

   vm->prepareFrame( cls->constructor().asFunction(), paramCount );
   vm->self() = inst;
   inst->gcMark( memPool->generation() );

   // also return self; we need to tell it to the VM
   vm->requestConstruct();

   if ( cls->hasInitEnter() )
   {
      vm->returnHandler( __call_init_enter );
   }
}

//=============================================================
// Tables declaration
//

void* NilCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // neg
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_nil_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable
};

void* IntCommOpsTable[] = {
   (void*) co_int_add,
   (void*) co_int_sub,
   (void*) co_int_mul,
   (void*) co_int_div,
   (void*) co_int_mod,
   (void*) co_int_pow,
   (void*) co_int_neg,

   // Inc/dec
   (void*) co_int_inc,
   (void*) co_int_dec,
   (void*) co_int_incpost,
   (void*) co_int_decpost,

   // cfr
   (void*) co_int_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable

};

void* NumCommOpsTable[] = {
   (void*) co_num_add,
   (void*) co_num_sub,
   (void*) co_num_mul,
   (void*) co_num_div,
   (void*) co_num_mod,
   (void*) co_num_pow,
   (void*) co_num_neg,

   // Inc/dec
   (void*) co_num_inc,
   (void*) co_num_dec,
   (void*) co_num_incpost,
   (void*) co_num_decpost,

   // cfr
   (void*) co_num_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable
};

void* RangeCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_range_compare,

   // set deep
   (void*) co_range_getindex,
   (void*) co_range_setindex,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable


};

void* BoolCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_bool_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable
};


void* LBindCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_lbind_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable
};

void* FuncCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_func_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_function
};


void* GCPTRCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_gcptr_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_generic_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable
};

void* StringCommOpsTable[] = {
   (void*) co_string_add,
   (void*) co_fail,
   (void*) co_string_mul,
   (void*) co_string_div,
   (void*) co_string_mod,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_string_compare,

   // set deep
   (void*) co_string_getindex,
   (void*) co_string_setindex,
   (void*) co_string_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_uncallable
};

void* ArrayCommOpsTable[] = {
   (void*) co_array_add,
   (void*) co_array_sub,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_array_compare,

   // set deep
   (void*) co_deep_getindex,
   (void*) co_deep_setindex,
   (void*) co_deep_getproperty,
   (void*) co_deep_setproperty,

   //call
   (void*) co_call_array
};

void* DictCommOpsTable[] = {
   (void*) co_dict_add,
   (void*) co_dict_sub,
   (void*) co_dict_mul,
   (void*) co_dict_div,
   (void*) co_dict_mod,
   (void*) co_dict_pow,
   (void*) co_dict_neg,

   // Inc/dec
   (void*) co_dict_inc,
   (void*) co_dict_dec,
   (void*) co_dict_incpost,
   (void*) co_dict_decpost,

   // cfr
   (void*) co_dict_compare,

   // set deep
   (void*) co_deep_getindex,
   (void*) co_deep_setindex,
   (void*) co_deep_getproperty,
   (void*) co_deep_setproperty,

   //call
   (void*) co_call_dict
};

void* ObjectCommOpsTable[] = {
   (void*) co_object_add,
   (void*) co_object_sub,
   (void*) co_object_mul,
   (void*) co_object_div,
   (void*) co_object_mod,
   (void*) co_object_pow,
   (void*) co_object_neg,

   // Inc/dec
   (void*) co_object_inc,
   (void*) co_object_dec,
   (void*) co_object_incpost,
   (void*) co_object_decpost,

   // cfr
   (void*) co_object_compare,

   // set deep
   (void*) co_deep_getindex,
   (void*) co_deep_setindex,
   (void*) co_deep_getproperty,
   (void*) co_deep_setproperty,

   //call
   (void*) co_call_object
};

void* MembufCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_membuf_compare,

   // set deep
   (void*) co_deep_getindex,
   (void*) co_deep_setindex,
   (void*) co_deep_getproperty,
   (void*) co_deep_setproperty,

   //call
   (void*) co_call_uncallable
};

void* ReferenceCommOpsTable[] = {
   (void*) co_ref_add,
   (void*) co_ref_sub,
   (void*) co_ref_mul,
   (void*) co_ref_div,
   (void*) co_ref_mod,
   (void*) co_ref_pow,
   (void*) co_ref_neg,

   (void*) co_ref_inc,
   (void*) co_ref_dec,
   (void*) co_ref_incpost,
   (void*) co_ref_decpost,

   // cfr
   (void*) co_ref_compare,

   // set deep
   (void*) co_ref_getindex,
   (void*) co_ref_setindex,
   (void*) co_ref_getproperty,
   (void*) co_ref_setproperty,

   //call
   (void*) co_call_reference
};

void* ClsMethodCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_clsmethod_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_classmeth_getproperty,
   (void*) co_fail,

   //call -- act as a class
   // this is possible as long as asMethodClass() === asClass()
   (void*) co_call_class

};

void* MethodCommOpsTable[] = {
   (void*) co_method_add,
   (void*) co_method_sub,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_method_compare,

   // set deep
   (void*) co_method_getindex,
   (void*) co_method_setindex,
   (void*) co_method_getproperty,
   (void*) co_fail,

   //call
   (void*) co_call_method

};

void* ClassCommOpsTable[] = {
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // Inc/dec
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_fail,

   // cfr
   (void*) co_class_compare,

   // set deep
   (void*) co_fail,
   (void*) co_fail,
   (void*) co_class_getproperty,
   (void*) co_class_setproperty,

   //call
   (void*) co_call_class

};


void* UnbCommOpsTable[] = {
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_fail_unb,

   // Inc/dec
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_fail_unb,

   // cfr
   (void*) co_unb_compare,

   // set deep
   (void*) co_fail_unb,
   (void*) co_fail_unb,
   (void*) co_generic_getproperty,
   (void*) co_fail_unb,

   //call
   (void*) co_fail_unb

};

CommOpsTable CommOpsDict[] =
{
   NilCommOpsTable,
   BoolCommOpsTable,
   IntCommOpsTable,
   NumCommOpsTable,
   RangeCommOpsTable,
   LBindCommOpsTable,
   FuncCommOpsTable,
   GCPTRCommOpsTable,
   StringCommOpsTable,
   ArrayCommOpsTable,
   DictCommOpsTable,
   ObjectCommOpsTable,
   MembufCommOpsTable,
   ReferenceCommOpsTable,
   ClsMethodCommOpsTable,
   MethodCommOpsTable,
   ClassCommOpsTable,
   UnbCommOpsTable
};

}

/* end of item_co.cpp */
