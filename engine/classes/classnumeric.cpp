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
#define SRC "engine/classes/classnumeric.cpp"


#include <falcon/classes/classnumeric.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/optoken.h>
#include <falcon/vmcontext.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/stderrors.h>

#include <falcon/stdhandlers.h>

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
         item->setNumeric( param->forceNumeric() );
      }
      else if( param->isString() )
      {
         numeric value;
         if( ! param->asString()->parseDouble( value ) )
         {
            throw new ParamError( ErrorParam( e_param_range, __LINE__, __FILE__ ).extra( "Not a number" ) );
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


void ClassNumeric::restore( VMContext* ctx, DataReader* dr ) const
{
   numeric value;
   dr->read( value );
   ctx->pushData( Item().setNumeric(value) );
}


void ClassNumeric::describe( void* instance, String& target, int, int  ) const 
{
   target.N(((Item*) instance)->asNumeric() );   
}

Class* ClassNumeric::getParent( const String& name ) const
{
   static Class* number = Engine::handlers()->numberClass();

   if( name == number->name() )
   {
      return number;
   }
   return 0;
}

bool ClassNumeric::isDerivedFrom( const Class* cls ) const
{
   static Class* number = Engine::handlers()->numberClass();
   return cls == number || cls == this;
}

void ClassNumeric::enumerateParents( ClassEnumerator& cb ) const
{
   static Class* number = Engine::handlers()->numberClass();
   cb(number);
}

void* ClassNumeric::getParentData( const Class* parent, void* data ) const
{
   static Class* number = Engine::handlers()->numberClass();

   if( parent == number )
   {
      return data;
   }
   return 0;
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


// ---------------------------------------------------------------------

void ClassNumeric::op_inc( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   iself->setNumeric(iself->asNumeric() + 1.0);
   ctx->stackResult( 1, iself->asNumeric() );
}


void ClassNumeric::op_dec( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   iself->setNumeric(iself->asNumeric() - 1.0);
   ctx->stackResult( 1, iself->asNumeric() );
}


void ClassNumeric::op_incpost( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   ctx->stackResult( 1, iself->asNumeric() );
   iself->setNumeric(iself->asNumeric() + 1.0);
}


void ClassNumeric::op_decpost( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   ctx->stackResult( 1, iself->asNumeric() );
   iself->setNumeric(iself->asNumeric() - 1.0);
}


void ClassNumeric::op_iter( VMContext* ctx, void* self ) const
{
   Item* item = (Item*)self;
   int64 value = (int64) item->asNumeric();
   if( value == 0 )
   {
      ctx->pushData(*item);
      ctx->topData().setBreak();
   }
   else if ( value < 0 )
   {
      ctx->pushData( Item((int64)-1) );
   }
   else
   {
      ctx->pushData( Item((int64)1) );
   }
}


void ClassNumeric::op_next( VMContext* ctx, void* self ) const
{
   Item* item = (Item*)self;
   int64 limit = (int64) item->asNumeric();

   int64 current = ctx->topData().asInteger();
   ctx->pushData(Item(current));
   if( current != limit )
   {
      // ask another loop
      ctx->topData().setDoubt();
      // prepare for next loop
      if ( current < 0 )
      {
         ctx->opcodeParam(1).setInteger(current-1);
      }
      else
      {
         ctx->opcodeParam(1).setInteger(current+1);
      }
   }
}


void ClassNumeric::op_compare( VMContext* ctx, void* ) const
{
   Item *op1, *op2;

   ctx->operands( op1, op2 );
   numeric iop1 = op1->asNumeric();

   if( op2->isInteger() )
   {
      numeric iop2 = (numeric) op2->asInteger();
      ctx->stackResult(2, iop1 > iop2 ? 1 : (iop1 < iop2 ? -1 : 0 ) );
      return;
   }
   else if (op2->isNumeric())
   {
      numeric iop2 = op2->asNumeric();
      ctx->stackResult(2, iop1 > iop2 ? 1 : (iop1 < iop2 ? -1 : 0 ) );
      return;
   }

   // we have no information about what an item might be here, but we can
   // order the items by type
   ctx->stackResult(2, (int64) op1->type() - op2->type() );
}


}

/* end of classnumeric.cpp */

