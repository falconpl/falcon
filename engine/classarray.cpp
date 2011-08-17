/*
   FALCON - The Falcon Programming Language.
   FILE: classarray.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 15:33:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classarray.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/accesserror.h>
#include <falcon/paramerror.h>
#include <falcon/itemarray.h>
#include <falcon/codeerror.h>

namespace Falcon {

ClassArray::ClassArray():
   Class("Array", FLC_CLASS_ID_ARRAY )
{
}


ClassArray::~ClassArray()
{
}


void ClassArray::dispose( void* self ) const
{
   ItemArray* f = static_cast<ItemArray*>(self);
   delete f;
}


void* ClassArray::clone( void* source ) const
{
   ItemArray* array = static_cast<ItemArray*>(source);
   return new ItemArray(*array);
}


void ClassArray::serialize( DataWriter*, void* ) const
{
   // todo
}


void* ClassArray::deserialize( DataReader* ) const
{
   //todo
   return 0;
}


void ClassArray::describe( void* instance, String& target, int maxDepth, int maxLen ) const
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


void ClassArray::op_create( VMContext* ctx, int pcount ) const
{
   static Collector* coll = Engine::instance()->collector();

   length_t count = 0;

   if ( pcount >= 1 )
   {
      Item* first = ctx->opcodeParams(pcount);
      if( ! first->isOrdinal() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__,__FILE__ ).extra( "[N]" ) );
      }
      count = first->forceInteger();
   }

   ctx->stackResult( pcount+1, FALCON_GC_STORE( coll, this, new ItemArray( count ) ) );
}


void ClassArray::op_getProperty( VMContext* ctx, void* self, const String& property ) const
{
   Class::op_getProperty( ctx, self, property );
}


void ClassArray::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *index, *arritem;
   ctx->operands( arritem, index );

   ItemArray& array = *static_cast<ItemArray*>(self);
   if (index->isOrdinal())
   {
      int64 v = index->forceInteger();
      if( v < 0 ) v = array.length() + v;
      if( v >= array.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("out of range") );
      }
      ctx->stackResult(2, array[v]);
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra("N") );
   }
}


void ClassArray::op_setIndex( VMContext* ctx, void* self ) const
{
   Item* value, *arritem, *index;
   ctx->operands( value, arritem, index );

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
      ctx->stackResult(3, *value);
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ) );
   }
}


void ClassArray::gcMark( void* self, uint32 mark ) const
{
   ItemArray& array = *static_cast<ItemArray*>(self);
   array.gcMark( mark );
}


void ClassArray::enumerateProperties( void*, Class::PropertyEnumerator& ) const
{
   // TODO array bindings?
}

void ClassArray::enumeratePV( void*, Class::PVEnumerator& ) const
{
   // TODO array bindings?
}

//=======================================================================
//

void ClassArray::op_add( VMContext* ctx, void* self ) const
{
   static Class* arrayClass = Engine::instance()->arrayClass();
   static Collector* coll = Engine::instance()->collector();
   
   ItemArray* array = static_cast<ItemArray*>(self);
   Item* op1, *op2;
   ctx->operands( op1, op2 );

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
    
   ctx->stackResult( 2, Item( FALCON_GC_STORE( coll, arrayClass, result ) ) );
}


void ClassArray::op_aadd( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>(self);
   Item* op1, *op2;
   ctx->operands( op1, op2 );

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
   ctx->popData();
}


void ClassArray::op_isTrue( VMContext* ctx, void* self) const
{
   ctx->stackResult(1, static_cast<ItemArray*>(self)->length() != 0 );
}


void ClassArray::op_toString( VMContext* ctx, void* self ) const
{
   String s;
   s.A("[Array of ").N(static_cast<ItemArray*>(self)->length()).A(" elements]");
   ctx->stackResult( 1, s );
}


void ClassArray::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   ItemArray* data = static_cast<ItemArray*>(self);
   length_t len = data->length();
   
   if( len == 0 )
   {
      Class::op_call( ctx, paramCount, self ); 
   }
   else 
   {
      Item* elems = data->elements();
      
      Class *cls;
      void* udata;
      elems->forceClassInst( cls, udata );
      if( udata == data)
      {
         // infinite expand loop.
         throw new CodeError( ErrorParam( e_call_loop, __LINE__, SRC )
            .origin( ErrorParam::e_orig_vm )
            );
      }
      
      // expand the array as parameters.
      ctx->insertData( paramCount+1, data->elements(), (int32) len, 1 );
      
      // invoke the call.
      cls->op_call( ctx, len + paramCount-1, udata );
   }   
}


void ClassArray::op_iter( VMContext* ctx, void* ) const
{
   // (seq)
   ctx->pushData( (int64) 0 );
}


void ClassArray::op_next( VMContext* ctx, void* instance ) const
{
   // (seq)(iter)
   Item& iter = ctx->opcodeParam(0);
   length_t pos = 0;
   
   ItemArray* arr = static_cast<ItemArray*>(instance);
   fassert( iter.isInteger() );
   pos = (length_t) iter.asInteger();
   if( pos >= arr->length() )
   {
      ctx->addDataSlot().setBreak();
   }
   else
   {
      
      Item& value = arr->at(pos++);
      value.copied();
      iter.setInteger( pos );
      ctx->pushData( value );
      if( pos >= arr->length() )
      {
         ctx->topData().setLast();
      }
   }
}



#if 0
void ClassArray::op_toString( VMContext* ctx, void* self ) const
   // If we're long 0, surrender.
   ItemArray* array = static_cast<ItemArray*>(self);

   if( array->length() == 0 )
   {
      ctx->stackResult(1, "[]");
      return;
   }
   VMContext* ctx = ctx->currentContext();

   // initialize the deep loop
   ctx->addDataSlot() = 0;  // local 0 -- counter
   ctx->addDataSlot().setString("[ ");  // local 1 -- currently elaborated string.

   // add an item for op_toString
   ctx->pushData(Item());

   m_toStringNextOp.apply(&m_toStringNextOp, vm);
}


ClassArray::ToStringNextOp::ToStringNextOp()
{
   apply = apply_;
}


void ClassArray::ToStringNextOp::apply_( const PStep* step, VMachine* vm )
{
   VMContext* ctx = ctx->currentContext();
   
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
         
         ctx->ifDeep( step );
         cls->op_toString( vm, inst );
         if( ctx->wentDeep() ) return;

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

/* end of classarray.cpp */
