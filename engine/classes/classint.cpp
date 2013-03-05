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

#undef SRC
#define SRC "engine/classes/classsint.cpp"

#include <falcon/classes/classint.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/optoken.h>
#include <falcon/vmcontext.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/stdhandlers.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/operanderror.h>

#include <math.h>

namespace Falcon {

ClassInt::ClassInt():
   Class( "Integer", FLC_ITEM_INT )
{
   m_bIsFlatInstance = true;
}


ClassInt::~ClassInt()
{
}


void ClassInt::dispose( void* ) const
{  
}


void* ClassInt::clone( void* source ) const
{
   return source;
}

void* ClassInt::createInstance() const
{
   // this is a flat class.
   return 0;
}


void ClassInt::store( VMContext*, DataWriter* dw, void* data ) const
{
   dw->write( static_cast<Item*>( data )->asInteger() );
}


void ClassInt::restore( VMContext* ctx, DataReader* dr ) const
{
   int64 value;
   dr->read( value );
   Item v;
   v.setInteger(value);
   ctx->pushData( v );
}


void ClassInt::describe( void* instance, String& target, int, int ) const
{
   target.N(((Item*) instance)->asInteger() );
}


Class* ClassInt::getParent( const String& name ) const
{
   static Class* number = Engine::handlers()->numberClass();

   if( name == number->name() )
   {
      return number;
   }
   return 0;
}

bool ClassInt::isDerivedFrom( const Class* cls ) const
{
   static Class* number = Engine::handlers()->numberClass();
   return cls == number;
}

void ClassInt::enumerateParents( ClassEnumerator& cb ) const
{
   static Class* number = Engine::handlers()->numberClass();
   cb(number);
}

void* ClassInt::getParentData( const Class* parent, void* data ) const
{
   static Class* number = Engine::handlers()->numberClass();

   if( parent == number )
   {
      Item* itm = static_cast<Item*>(data);
      itm->setInteger((int64) itm->asNumeric());
      return data;
   }
   return 0;
}

//=======================================================================
//

bool ClassInt::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   Item* item = static_cast<Item*>(instance);
   
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
           item->setInteger( value );
         }
      }
      else
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__, __FILE__ ).extra( "(N|S)" ) );
      }
   }
   else
   {
      item->setInteger(0);
   }
   
   return false;
}


void ClassInt::op_isTrue( VMContext* ctx, void* self ) const
{
   Item* iself = static_cast<Item*>(self);
   ctx->topData().setBoolean( iself->asInteger() != 0 );
}

void ClassInt::op_toString( VMContext* ctx, void* self ) const
{
   Item* iself = static_cast<Item*>(self);
   String* s = new String;
   s->N( iself->asInteger() );   
   ctx->topData() = FALCON_GC_HANDLE(s); // will garbage S
}


void ClassInt::op_add( VMContext* ctx, void* self ) const 
{    
   Item *iself, *op2;   
   iself = (Item*)self;
   op2 = &ctx->topData();
   
   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() + op2->asInteger() );
         break;
   
      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() + op2->asNumeric() );
         break;
         
      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "+" ) );
   }
}


void ClassInt::op_sub( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() - op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() - op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "-" ) );
   }
}


void ClassInt::op_mul( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() * op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() * op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "*" ) );
   }
}

void ClassInt::op_div( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() / op2->forceNumeric() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() / op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "/" ) );
   }
}


void ClassInt::op_mod( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() % op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() % op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "%" ) );
   }
}


void ClassInt::op_pow( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, (int64)pow( (long double)iself->asInteger(), (long double)op2->asInteger() ) );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, (double)pow( (long double)iself->asInteger(), (long double)op2->asNumeric() ) );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "**" ) );
   }
}


void ClassInt::op_shr( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() >> op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() >> op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( ">>" ) );
   }
}

void ClassInt::op_shl( VMContext* ctx, void* self ) const
{
    Item *iself, *op2;
    iself = (Item*)self;
    op2 = &ctx->topData();

    switch( op2->type() )
    {
       case FLC_ITEM_INT:
          ctx->stackResult(2, iself->asInteger() << op2->asInteger() );
          break;

       case FLC_ITEM_NUM:
          ctx->stackResult(2, iself->asInteger() << op2->forceInteger() );
          break;

       default:
          throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
                .origin( ErrorParam::e_orig_vm )
                .extra( "<<" ) );
    }
 }

void ClassInt::op_aadd( VMContext* ctx, void*self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() + op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() + op2->asNumeric() );
         break;


      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "+=" ) );
   }
}

void ClassInt::op_asub( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() - op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() - op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "-=" ) );
   }
}


void ClassInt::op_amul( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() * op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() * op2->asNumeric() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "*=" ) );
   }
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


void ClassInt::op_amod( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() % op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() % op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "%=" ) );
   }
}


void ClassInt::op_apow( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, (int64)pow( (long double)iself->asInteger(), (long double)op2->asInteger() ) );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, (double)pow( (long double)iself->asInteger(), (long double)op2->asNumeric() ) );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "**=" ) );
   }
}


void ClassInt::op_ashr( VMContext* ctx, void* self ) const
{
   Item *iself, *op2;
   iself = (Item*)self;
   op2 = &ctx->topData();

   switch( op2->type() )
   {
      case FLC_ITEM_INT:
         ctx->stackResult(2, iself->asInteger() >> op2->asInteger() );
         break;

      case FLC_ITEM_NUM:
         ctx->stackResult(2, iself->asInteger() >> op2->forceInteger() );
         break;

      default:
         throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( ">>=" ) );
   }
}

void ClassInt::op_ashl( VMContext* ctx, void* self ) const
{
    Item *iself, *op2;
    iself = (Item*)self;
    op2 = &ctx->topData();

    switch( op2->type() )
    {
       case FLC_ITEM_INT:
          ctx->stackResult(2, iself->asInteger() << op2->asInteger() );
          break;

       case FLC_ITEM_NUM:
          ctx->stackResult(2, iself->asInteger() << op2->forceInteger() );
          break;

       default:
          throw new OperandError( ErrorParam( e_invalid_op, __LINE__, SRC )
                .origin( ErrorParam::e_orig_vm )
                .extra( "<<=" ) );
    }
 }


// -------- Helper functions to increment and decrement ClassInt -----
// ---------------------------------------------------------------------

void ClassInt::op_inc( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   iself->setInteger(iself->asInteger()+1);
   ctx->stackResult( 1, iself->asInteger() );
}


void ClassInt::op_dec( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   iself->setInteger(iself->asInteger() - 1);
   ctx->stackResult( 1, iself->asInteger() );
}


void ClassInt::op_incpost( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   ctx->stackResult( 1, iself->asInteger() );
   iself->setInteger(iself->asInteger() + 1);
}


void ClassInt::op_decpost( VMContext* ctx, void* self ) const
{
   Item *iself = (Item*)self;
   ctx->stackResult( 1, iself->asInteger() );
   iself->setInteger(iself->asInteger() - 1);
}

}

/* end of classint.cpp */

