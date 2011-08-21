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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_add( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_sub( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_mul( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_div( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_mod( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_pow( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_shr( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

       case FLC_ITEM_USER:
          if( ctx->topData().deuser() )
          {
             op_shl( ctx, self );
             break;
          }
          // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_aadd( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_asub( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_amul( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_adiv( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_amod( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_apow( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

      case FLC_ITEM_USER:
         if( ctx->topData().deuser() )
         {
            op_ashr( ctx, self );
            break;
         }
         // else, fallthrough and raise the error.

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

       case FLC_ITEM_USER:
          if( ctx->topData().deuser() )
          {
             op_ashl( ctx, self );
             break;
          }
          // else, fallthrough and raise the error.

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

