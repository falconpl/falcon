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
#include <falcon/accesserror.h>
#include <falcon/paramerror.h>
#include <falcon/itemarray.h>

namespace Falcon {

CoreArray::CoreArray():
   Class("Array", FLC_CLASS_ID_ARRAY )
{
}


CoreArray::~CoreArray()
{
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


void CoreArray::serialize( DataWriter*, void* ) const
{
   // todo
}


void* CoreArray::deserialize( DataReader* ) const
{
   //todo
   return 0;
}


void CoreArray::describe( void* instance, String& target, int maxDepth, int maxLen ) const
{
   ItemArray* arr = static_cast<ItemArray*>(instance);
   target += "[";

   String temp;
   for( length_t pos = 0; pos < arr->length(); ++ pos )
   {
      if( pos > 0 )
      {
         target += ", ";
      }
      
      Class* cls;
      void* inst;
      arr->at(pos).forceClassInst(cls, inst);
      temp.size(0);
      cls->describe( inst, temp, maxDepth - 1, maxLen );

      target += temp;
   }

   target += "]";
}


void CoreArray::op_create( VMachine* vm, int pcount ) const
{
   static Collector* coll = Engine::instance()->collector();

   length_t count = 0;

   if ( pcount >= 1 )
   {
      Item* first = vm->currentContext()->opcodeParams(pcount);
      if( ! first->isOrdinal() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__,__FILE__ ).extra( "[N]" ) );
      }
      count = first->forceInteger();
   }

   vm->stackResult( pcount+1, FALCON_GC_STORE( coll, this, new ItemArray( count ) ) );
}


void CoreArray::op_getProperty( VMachine* vm, void* self, const String& property ) const
{
   Class::op_getProperty( vm, self, property );
}

void CoreArray::op_getIndex( VMachine* vm, void* self ) const
{
   Item *index, *arritem;
   vm->operands( arritem, index );

   ItemArray& array = *static_cast<ItemArray*>(self);
   if (index->isOrdinal())
   {
      int64 v = index->forceInteger();
      if( v < 0 ) v = array.length() + v;
      if( v >= array.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("out of range") );
      }
      vm->stackResult(2, array[v]);
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("N") );
   }
}


void CoreArray::op_setIndex( VMachine* vm, void* self ) const
{
   Item* value, *index, *arritem;
   vm->operands( value, arritem, index );

   ItemArray& array = *static_cast<ItemArray*>(self);
   if (index->isOrdinal())
   {
      int64 v = index->forceInteger();
      if( v < 0 ) v = array.length() + v;
      if( v >= array.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("out of range") );
      }
      // the value is copied here.
      value->copied(true);
      array[v] = *value;
      vm->stackResult(3, *value);
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ) );
   }
}


void CoreArray::gcMark( void* self, uint32 mark ) const
{
   ItemArray& array = *static_cast<ItemArray*>(self);
   array.gcMark( mark );
}


void CoreArray::enumerateProperties( void*, PropertyEnumerator& ) const
{
   // TODO array bindings?
}


//=======================================================================
//

void CoreArray::op_add( VMachine *vm, void* self ) const
{
   static Class* arrayClass = Engine::instance()->arrayClass();
   static Collector* coll = Engine::instance()->collector();
   
   ItemArray* array = static_cast<ItemArray*>(self);
   Item* op1, *op2;
   vm->operands( op1, op2 );

   Class* cls;
   void* inst;
   
   ItemArray *result = new ItemArray;
    
   // a basic type?
   if( ! op2->asClassInst( cls, inst ) || cls->typeID() != typeID() )
   {
      result->reserve( array->length() + 1 );
      result->merge(*array);
      
      op2->copied(true);
      result->append(*op2);
   }
   else {
      // it's an array!
      ItemArray* other = static_cast<ItemArray*>(inst);
      result->reserve( array->length() + other->length() );
      result->merge( *array );
      result->merge( *other );
   }
    
   vm->stackResult( 2, Item( FALCON_GC_STORE( coll, arrayClass, result ) ) );
}


void CoreArray::op_aadd( VMachine *vm, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>(self);
   Item* op1, *op2;
   vm->operands( op1, op2 );

   Class* cls;
   void* inst;
   
   // a basic type?
   if( ! op2->asClassInst( cls, inst ) || cls->typeID() != typeID() )
   {
      op2->copied(true);
      array->append(*op2);
   }
   else {
      // it's an array!
      ItemArray* other = static_cast<ItemArray*>(inst);
      array->merge( *other );
   }

   // just remove the topmost item,
   vm->currentContext()->popData();
}


void CoreArray::op_isTrue( VMachine *vm, void* self) const
{
   vm->stackResult(1, static_cast<ItemArray*>(self)->length() != 0 );
}


void CoreArray::op_toString( VMachine *vm, void* self ) const
{
   String s;
   s.A("[Array of ").N(static_cast<ItemArray*>(self)->length()).A(" elements]");
   vm->stackResult( 1, s );
}

#if 0
void CoreArray::op_toString( VMachine *vm, void* self ) const
   // If we're long 0, surrender.
   ItemArray* array = static_cast<ItemArray*>(self);

   if( array->length() == 0 )
   {
      vm->stackResult(1, "[]");
      return;
   }
   VMContext* ctx = vm->currentContext();

   // initialize the deep loop
   ctx->addDataSlot() = 0;  // local 0 -- counter
   ctx->addDataSlot().setString("[ ");  // local 1 -- currently elaborated string.

   // add an item for op_toString
   ctx->pushData(Item());

   m_toStringNextOp.apply(&m_toStringNextOp, vm);
}


CoreArray::ToStringNextOp::ToStringNextOp()
{
   apply = apply_;
}


void CoreArray::ToStringNextOp::apply_( const PStep* step, VMachine* vm )
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
      // in this case, get the top data -- now converted by the deep call.
      fassert( ctx->topData().asString() )
      myStr->append(*ctx->topData().asString());
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
         ctx->topData() = item;
         
         vm->ifDeep( step );
         cls->op_toString( vm, inst );
         if( vm->wentDeep() ) return;

         // pfew, not deep! just go on as usual
         fassert(ctx->topData().isString());
         myStr->append( *ctx->topData().asString() );
      }
   }

   // we're done, put our string in A.
   myStr->append(" ]");
   ctx->popData();
   ctx->topData() = myStr->garbage();
}
#endif

}

/* end of corearray.cpp */