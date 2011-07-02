/*
   FALCON - The Falcon Programming Language.
   FILE: classint.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classint.h>
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

ClassInt::ClassInt():
   Class( "Integer", FLC_ITEM_INT )
{
}


ClassInt::~ClassInt()
{
}


void ClassInt::dispose( void* self ) const
{
   Item* data = (Item*) self;
   delete data;
}


void* ClassInt::clone( void* source ) const
{
   Item* ptr = new Item;
   *ptr = *(Item*) source;
   return ptr;
}


void ClassInt::serialize( DataWriter *stream, void *self ) const
{
   
   int64 value = static_cast< Item* >( self )->asInteger();

   stream->write( value );

}


void* ClassInt::deserialize( DataReader *stream ) const
{
   
   int64 value;

   stream->read( value );

   return new Item( value );

}

void ClassInt::describe( void* instance, String& target, int, int ) const
{
   target.N(((Item*) instance)->asInteger() );
}

//=======================================================================
//

void ClassInt::op_create( VMContext* ctx, int pcount ) const
{
   if( pcount > 0 )
   {
      Item* param = ctx->opcodeParams(pcount);
      if( param->isOrdinal() )
      {
         ctx->stackResult( pcount + 1, param->forceInteger() );
      }
      else if( param->isString() )
      {
         int64 value;
         if( ! param->asString()->parseInt( value ) )
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
      ctx->stackResult( pcount + 1, Item( (int64) 0 ) );
   }
}


void ClassInt::op_isTrue( VMContext* ctx, void* ) const
{
   Item* iself;
   OpToken token( ctx, iself );
   token.exit( iself->asInteger() != 0 );
}

void ClassInt::op_toString( VMContext* ctx, void* ) const
{
   Item* iself;
   OpToken token( ctx, iself );
   String s;
   token.exit( s.N(iself->asInteger()) );
}

void ClassInt::op_add( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() )
   {

      token.exit( self->asInteger() + op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      token.exit( self->forceNumeric() + op2->asNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}

void ClassInt::op_sub( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() )
   {

      token.exit( self->asInteger() - op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      token.exit( self->forceNumeric() - op2->asNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassInt::op_mul( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() )
   {

      token.exit( self->asInteger() * op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      token.exit( self->forceNumeric() * op2->asNumeric() );

   }
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassInt::op_div( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() )
   {

      token.exit( self->asInteger() / op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      token.exit( self->forceNumeric() / op2->asNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassInt::op_mod( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() )
   {

      token.exit( self->asInteger() % op2->asInteger() );

   }
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
    
}


void ClassInt::op_pow( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   OpToken token( ctx, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_NUM )
   {

      token.exit( pow( self->forceNumeric(), op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void ClassInt::op_aadd( VMContext* ctx, void*) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() ) 
   {

      self->setInteger( self->asInteger() + op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      self->setNumeric( self->forceNumeric() + op2->asNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}

void ClassInt::op_asub( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() ) 
   {

      self->setInteger( self->asInteger() - op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      self->setNumeric( self->forceNumeric() - op2->asNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassInt::op_amul( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() ) 
   {

      self->setInteger( self->asInteger() * op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      self->setNumeric( self->forceNumeric() * op2->asNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassInt::op_adiv( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() ) 
   {

      self->setInteger( self->asInteger() / op2->asInteger() );

   }

   else if( op2->type() == FLC_ITEM_NUM )
   {

      self->setNumeric( self->forceNumeric() / op2->asNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassInt::op_amod( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() ) 
   {

      self->setInteger( self->asInteger() % op2->asInteger() );

   }
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassInt::op_apow( VMContext* ctx, void* ) const {
    
   Item *self, *op2;

   ctx->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_NUM ) 
   {

      self->setNumeric( pow( self->forceNumeric() , op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   ctx->popData(); // Put self on the top of the stack
    
}


void ClassInt::op_inc( VMContext* ctx, void* ) const
{    
   Item *self;
   ctx->operands( self );
   self->setInteger( self->asInteger() + 1 );
}


void ClassInt::op_dec( VMContext* ctx, void*) const
{    
   Item *self;
   ctx->operands( self );
   self->setInteger( self->asInteger() + 1 );
}


void ClassInt::op_incpost( VMContext*, void* ) const
{    
   // TODO   
}


void ClassInt::op_decpost( VMContext*, void* ) const
{    
   // TODO   
}

}

/* end of classint.cpp */
