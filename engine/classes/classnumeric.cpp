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

#undef SRC
#define SRC "engine/classes/classsnumeric.cpp"


#include <falcon/classes/classnumeric.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/optoken.h>
#include <falcon/vmcontext.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/operanderror.h>

#include <math.h>

namespace Falcon {
    

ClassNumeric::ClassNumeric() : 
   Class( "Numeric", FLC_ITEM_NUM ) 
{ 
   m_bIsFlatInstance = true; 
}


ClassNumeric::~ClassNumeric() 
{ 
}

bool ClassNumeric::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   Item* item = static_cast<Item*>(instance);
   
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
             item->setNumeric( value );
         }
      }
      else
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__, __FILE__ ).extra( "(N|S)" ) );
      }
   }
   else
   {
      item->setNumeric(0.0);
   }
   
   return false;
}


void ClassNumeric::dispose( void* ) const
{  
}


void *ClassNumeric::clone( void* source ) const 
{
   return source;
}

void *ClassNumeric::createInstance() const 
{
   return 0;
}

void ClassNumeric::store( VMContext*, DataWriter* dw, void* data ) const
{
   dw->write( static_cast<Item*>( data )->asNumeric() );
}


void ClassNumeric::restore( VMContext* , DataReader* dr, void*& data ) const
{
   int64 value;
   dr->read( value );
   static_cast<Item*>( data )->setNumeric(value);
}


void ClassNumeric::describe( void* instance, String& target, int, int  ) const 
{
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

void ClassNumeric::op_add( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() + op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() + op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "+" ) );
   }
}


void ClassNumeric::op_sub( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() - op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() - op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "-" ) );
   }
}


void ClassNumeric::op_mul( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() * op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() * op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "*" ) );
   }
}

void ClassNumeric::op_div( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() / op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() / op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "/" ) );
   }
}


void ClassNumeric::op_mod( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->forceInteger() % op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->forceInteger() % op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "%" ) );
   }
}


void ClassNumeric::op_pow( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, (double)pow( (long double)iself->asNumeric(), (long double)op2->asInteger() ) );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, pow( iself->asNumeric(), op2->asNumeric() ) );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "**" ) );
   }
}


void ClassNumeric::op_shr( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->forceInteger() >> op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->forceInteger() >> op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( ">>" ) );
   }
}

void ClassNumeric::op_shl( VMContext* ctx, void* self ) const
{
    Item *iself, *op2;
    iself = (Item*)self;
    op2 = &ctx->topData();

    switch( op2->type() )
    {
       case FLC_ITEM_INT:
          ctx->stackResult(2, iself->forceInteger() << op2->asInteger() );
          break;

       case FLC_ITEM_NUM:
          ctx->stackResult(2, iself->forceInteger() << op2->forceInteger() );
          break;

       default:
          throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
                .origin( ErrorParam::e_orig_vm )
                .extra( "<<" ) );
    }
 }

void ClassNumeric::op_aadd( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() + op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() + op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "+=" ) );
   }
}


void ClassNumeric::op_asub( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() - op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() - op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "-=" ) );
   }
}


void ClassNumeric::op_amul( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() * op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() * op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "*=" ) );
   }
}


void ClassNumeric::op_adiv( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asNumeric() / op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asNumeric() / op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "/=" ) );
   }
}


void ClassNumeric::op_amod( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->forceInteger() % op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->forceInteger() % op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "%=" ) );
   }
}


void ClassNumeric::op_apow( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, (double)pow( (long double)iself->asNumeric(), (long double)op2->asInteger() ) );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, pow( iself->asNumeric(), op2->asNumeric() ) );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "**=" ) );
   }
}


void ClassNumeric::op_ashr( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->forceInteger() >> op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->forceInteger() >> op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( ">>=" ) );
   }
}

void ClassNumeric::op_ashl( VMContext* ctx, void* self ) const
{
    Item *iself, *op2;
    iself = (Item*)self;
    op2 = &ctx->topData();

    switch( op2->type() )
    {
       case FLC_ITEM_INT:
          ctx->stackResult(2, iself->forceInteger() << op2->asInteger() );
          break;

       case FLC_ITEM_NUM:
          ctx->stackResult(2, iself->forceInteger() << op2->forceInteger() );
          break;

       default:
          throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
                .origin( ErrorParam::e_orig_vm )
                .extra( "<<=" ) );
    }
 }


// -------- Helper functions to increment and decrement ClassNumeric -----

inline void increment( VMContext *ctx, void *self )
{
   Item *iself = (Item*)self;
   ctx->stackResult( 1, iself->asInteger() + 1.0 );
}

inline void decrement( VMContext *ctx, void *self )
{
   Item *iself = (Item*)self;
   ctx->stackResult( 1, iself->asInteger() - 1.0 );
}

// ---------------------------------------------------------------------

void ClassNumeric::op_inc( VMContext* ctx, void* self ) const
{
   increment( ctx, self );
}


void ClassNumeric::op_dec( VMContext* ctx, void* self ) const
{
   decrement( ctx, self );
}


void ClassNumeric::op_incpost( VMContext* ctx, void* self ) const
{
   increment( ctx, self );
}


void ClassNumeric::op_decpost( VMContext* ctx, void* self ) const
{
   decrement( ctx, self );
}

}

/* end of classnumeric.cpp */

