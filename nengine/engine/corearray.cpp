/*
   FALCON - The Falcon Programming Language.
   FILE: corearray.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 15:33:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/corearray.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>

#include "falcon/itemarray.h"

namespace Falcon {

CoreArray::CoreArray():
   Class("Array", FLC_CLASS_ID_ARRAY )
{
}


CoreArray::~CoreArray()
{
}


void* CoreArray::create(void* creationParams ) const
{
   // Shall we copy a creator?
   cpars* cp = static_cast<cpars*>(creationParams);
   if( cp->m_other != 0 )
   {
      if ( cp->m_bCopy )
         return new ItemArray( *cp->m_other );
      else
         return cp->m_other;
   }

   // -- or shall we just generate a new array?
   return new ItemArray( cp->m_prealloc );
}


void CoreArray::dispose( void* self ) const
{
   ItemArray* f = static_cast<ItemArray*>(self);
   delete f;
}


void* CoreArray::clone( void* source ) const
{
   ItemArray* array = static_cast<ItemArray*>(source);
   return new ItemArray(*array);
}


void CoreArray::serialize( DataWriter* stream, void* self ) const
{
   // todo
}


void* CoreArray::deserialize( DataReader* stream ) const
{
   //todo
}

void CoreArray::describe( void* instance, String& target ) const
{
   ItemArray* arr = static_cast<ItemArray*>(instance);
   target = String("[Array of ").N(arr->length()).A(" elements]");
}


bool CoreArray::hasProperty( void* self, const String& prop ) const
{
   if( prop == "len" ) return true;
   return false;
}

bool CoreArray::getProperty( void* self, const String& property, Item& value ) const
{
   if( property == "len" )
   {
      value.setInteger(static_cast<ItemArray*>(self)->length());
      return true;
   }

   return false;
}

bool CoreArray::getIndex( void* self, const Item& index, Item& value ) const
{
   ItemArray& array = *static_cast<ItemArray*>(self);
   if (index.isOrdinal())
   {
      int64 v = index.forceInteger();
      if( v < 0 ) v = array.length() + v;
      if( v >= array.length() ) return false;
      value = array[v];
      return true;
   }

   return false;
}

bool CoreArray::setIndex( void* self, const Item& index, const Item& value ) const
{
   ItemArray& array = *static_cast<ItemArray*>(self);
   if (index.isOrdinal())
   {
      int64 v = index.forceInteger();
      if( v < 0 ) v = array.length() + v;
      if( v >= array.length() ) return false;
      array[v] = value;
      return true;
   }

   return false;
}


void CoreArray::gcMark( void* self, uint32 mark ) const
{
   ItemArray& array = *static_cast<ItemArray*>(self);
   for ( size_t i = 0; i < array.length(); ++i )
   {
      Item& tgt = array.at(i);
      if( tgt.isDeep() )
      {
         tgt.asDeepClass()->gcMark(tgt.asDeepInst(), mark);
      }
   }
}

void CoreArray::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   cb("len", true);
}


//=======================================================================
//

void CoreArray::op_add( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   ItemArray* array = static_cast<ItemArray*>(self);

   Class* cls;
   void* inst;
   // a basic type?
   if( ! op2.asClassInst( cls, inst ) || cls->typeID() != typeID() )
   {
      array->append(op2);
      return;
   }

   // it's an array!
   ItemArray* other = static_cast<ItemArray*>(inst);
   array->change( *other, array->length(), 0 );
}

void CoreArray::op_isTrue( VMachine *vm, void* self, Item& target ) const
{
   target.setBoolean( static_cast<ItemArray*>(self)->length() != 0 );
}

void CoreArray::op_toString( VMachine *vm, void* self, Item& target ) const
{
   // If we're long 0, surrender.
   ItemArray* array = static_cast<ItemArray*>(self);

   if( array->length() == 0 )
   {
      target.setString("[]");
      return;
   }
   VMContext* ctx = vm->currentContext();

   // initialize the deep loop
   ctx->addDataSlot() = 0;  // local 0 -- counter
   ctx->addDataSlot().setString("[ ");  // local 1 -- currently elaborated string.
   m_toStringNextOp.apply(&m_toStringNextOp, vm);
}


CoreArray::ToStringNextOp::ToStringNextOp()
{
   apply = apply_;
}

void CoreArray::ToStringNextOp::apply_( const PStep*step, VMachine* vm )
{
   VMContext* ctx = vm->currentContext();
   
   // array in self
   ItemArray* array = static_cast<ItemArray*>(ctx->self().asInst());

   // loop counter in 0
   length_t i = (length_t) ctx->local(0)->asInteger();
   
   // string in 1
   String* myStr = ctx->local(1)->asString();
   fassert( myStr != 0 )

   // have we been called from deep?
   if( i > 0 )
   {
      // in this case, get A and add it to our string.
      fassert( vm->regA().asString() )
      myStr->append(*vm->regA().asString());
   }

   // continue with the stringification loop
   for( ; i < array->length(); ++i )
   {
      if( i > 0 )
      {
         myStr->append( ", " );
      }

      Class* cls;
      void* inst;
      Item& item = array->at(i);
      
      // if it's a simple item, add it's base representation
      if( ! item.asClassInst( cls, inst ) )
      {
         myStr->append( item.describe() );
      }
      else
      {
         // Not simple; we may be deep -- so prepare
         ctx->local(0)->setInteger(i+1);
         vm->ifDeep( step );
         cls->op_toString( vm, inst, vm->regA() );
         if( vm->wentDeep() ) return;

         // pfew, not deep! just go on as usual
         fassert(vm->regA().isString());
         myStr->append( *vm->regA().asString() );
      }
   }

   // we're done, put our string in A.
   myStr->append(" ]");
   vm->regA() = myStr;
}

}

/* end of corearray.cpp */
