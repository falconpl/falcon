/*
 FALCON - The Falcon Programming Language.
 FILE: classnumeric.cpp
 
 Function object handler.
 -------------------------------------------------------------------
 Author: Francesco Magliocca
 Begin: Sat, 11 Jun 2011 22:00:05 +0200
 
 -------------------------------------------------------------------
 (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#include <falcon/classnumeric.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/optoken.h>
#include <falcon/vmcontext.h>
#include <falcon/operanderror.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/paramerror.h>
#include <math.h>

namespace Falcon {
    

ClassNumeric::ClassNumeric() : Class( "Numeric", FLC_ITEM_NUM ) { }

ClassNumeric::~ClassNumeric() { }

void ClassNumeric::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount > 0 )
   {
      Item* param = ctx->opcodeParams(pcount);
      
      if( param->isOrdinal() )
      {
         ctx->stackResult( pcount + 1, param->forceNumeric() );
      }
      else if( param->isString() )
      {
         numeric value;
         if( ! param->asString()->parseDouble( value ) )
         {
            throw new ParamError( ErrorParam( e_param_range, __LINE__, __FILE__ ).extra( "Not an integer" ) );
         }
         else
         {
            ctx->stackResult( pcount + 1, value );
         }
      }
      else
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__, __FILE__ ).extra( "(N|S)" ) );
      }
   }
   else
   {
      ctx->stackResult( pcount + 1, Item( 0.0 ) );
   }
}


void ClassNumeric::dispose( void *self ) const {
    
   Item *data = (Item*)self;
    
   delete data;
    
}

void *ClassNumeric::clone( void *source ) const {
    
   Item *result = new Item;
    
   *result = *static_cast<Item*>( source );
    
   return result;
    
}

void ClassNumeric::serialize( DataWriter *stream, void *self ) const {
    
   numeric value = static_cast< Item* >( self )->asNumeric();

   stream->write( value );
    
}


void* ClassNumeric::deserialize( DataReader *stream ) const {
    
   numeric value;

   stream->read( value );

   return new Item( value );
    
}

void ClassNumeric::describe( void* instance, String& target, int, int  ) const {
    
   target.N(((Item*) instance)->asNumeric() );
    
}

// ================================================================


void ClassNumeric::op_isTrue( VMContext* ctx, void* ) const
{
   Item* iself;
   OpToken token( ctx, iself );
   token.exit( iself->asNumeric() != 0 );
}

void ClassNumeric::op_toString( VMContext* ctx, void* ) const
{
   Item* iself;
   OpToken token( ctx, iself );
   String s;
   token.exit( s.N(iself->asNumeric()) );
}

void ClassNumeric::op_add( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() + op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}

void ClassNumeric::op_sub( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() - op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassNumeric::op_mul( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() * op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassNumeric::op_div( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() / op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassNumeric::op_pow( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {
      token.exit( pow( self->asNumeric(), op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassNumeric::op_aadd( VMContext* ctx, void*) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() + op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}

void ClassNumeric::op_asub( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() - op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassNumeric::op_amul( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() * op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassNumeric::op_adiv( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() / op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassNumeric::op_apow( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( pow( self->asNumeric(), op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassNumeric::op_inc(VMContext* ctx, void* ) const
{
   Item *self;
   ctx->operands( self );
   self->setNumeric( self->asNumeric() + 1.0 );
    
}


void ClassNumeric::op_dec(VMContext* ctx, void*) const
{    
   Item *self;
   ctx->operands( self );
   self->setNumeric( self->asNumeric() - 1.0 );
}


void ClassNumeric::op_incpost(VMContext*, void* ) const
{
   // TODO   
}


void ClassNumeric::op_decpost(VMContext*, void* ) const
{
   // TODO   
}

}

/* end of classnumeric.cpp */
