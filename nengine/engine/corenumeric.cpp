/*
 FALCON - The Falcon Programming Language.
 FILE: corenumeric.cpp
 
 Function object handler.
 -------------------------------------------------------------------
 Author: Francesco Magliocca
 Begin: Sat, 11 Jun 2011 22:00:05 +0200
 
 -------------------------------------------------------------------
 (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#include <falcon/corenumeric.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/optoken.h>
#include <falcon/vm.h>
#include <falcon/operanderror.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/paramerror.h>

#include <math.h>

namespace Falcon {
    

CoreNumeric::CoreNumeric() : Class( "Numeric", FLC_ITEM_NUM ) { }

CoreNumeric::~CoreNumeric() { }

void CoreNumeric::op_create( VMachine *vm, int pcount ) const
{
   if( pcount > 0 )
   {
      Item* param = vm->currentContext()->opcodeParams(pcount);
      if( param->isOrdinal() )
      {
         vm->stackResult( pcount + 1, param->forceNumeric() );
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
            vm->stackResult( pcount + 1, value );
         }
      }
      else
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__, __FILE__ ).extra( "(N|S)" ) );
      }
   }
   else
   {
      vm->stackResult( pcount + 1, Item( 0.0 ) );
   }
}


void CoreNumeric::dispose( void *self ) const {
    
   Item *data = (Item*)self;
    
   delete data;
    
}

void *CoreNumeric::clone( void *source ) const {
    
   Item *result = new Item;
    
   *result = *static_cast<Item*>( source );
    
   return result;
    
}

void CoreNumeric::serialize( DataWriter *stream, void *self ) const {
    
   numeric value = static_cast< Item* >( self )->asNumeric();

   stream->write( value );
    
}


void* CoreNumeric::deserialize( DataReader *stream ) const {
    
   numeric value;

   stream->read( value );

   return new Item( value );
    
}

void CoreNumeric::describe( void* instance, String& target, int, int  ) const {
    
   target.N(((Item*) instance)->asNumeric() );
    
}

// ================================================================


void CoreNumeric::op_isTrue( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   token.exit( iself->asNumeric() != 0 );
}

void CoreNumeric::op_toString( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   String s;
   token.exit( s.N(iself->asNumeric()) );
}

void CoreNumeric::op_add( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() + op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}

void CoreNumeric::op_sub( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() - op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void CoreNumeric::op_mul( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() * op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void CoreNumeric::op_div( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {

      token.exit( self->asNumeric() / op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void CoreNumeric::op_pow( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT )
   {
      token.exit( pow( self->asNumeric(), op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void CoreNumeric::op_aadd( VMachine *vm, void*) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() + op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}

void CoreNumeric::op_asub( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() - op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreNumeric::op_amul( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() * op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreNumeric::op_adiv( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( self->asNumeric() / op2->forceNumeric() );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreNumeric::op_apow( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_INT ) 
   {

      self->setNumeric( pow( self->asNumeric(), op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreNumeric::op_inc(VMachine *vm, void* ) const {
    
   Item *self;

   vm->operands( self );

   self->setInteger( self->asInteger() + 1.0 );
    
}


void CoreNumeric::op_dec(VMachine *vm, void*) const {
    
   Item *self;

   vm->operands( self );

   self->setInteger( self->asInteger() - 1.0 );
    
}


void CoreNumeric::op_incpost(VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_decpost(VMachine *, void* ) const {
    
   // TODO
    
}


}
