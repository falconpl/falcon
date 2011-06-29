/*
   FALCON - The Falcon Programming Language.
   FILE: coreint.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/coreint.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/optoken.h>
#include <falcon/vm.h>
#include <falcon/operanderror.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <math.h>

namespace Falcon {

CoreInt::CoreInt():
   Class( "Integer", FLC_ITEM_INT )
{
}


CoreInt::~CoreInt()
{
}


void* CoreInt::create(void* creationParams ) const
{
   Item* ptr = new Item;
   *ptr = *( (int64*)creationParams );
   return ptr;
}


void CoreInt::dispose( void* self ) const
{
   Item* data = (Item*) self;
   delete data;
}


void* CoreInt::clone( void* source ) const
{
   Item* ptr = new Item;
   *ptr = *(Item*) source;
   return ptr;
}


void CoreInt::serialize( DataWriter *stream, void *self ) const
{
   
   int64 value = static_cast< Item* >( self )->asInteger();

   stream->write( value );

}


void* CoreInt::deserialize( DataReader *stream ) const
{
   
   int64 value;

   stream->read( value );

   return new Item( value );

}

void CoreInt::describe( void* instance, String& target, int, int ) const
{
   target.N(((Item*) instance)->asInteger() );
}

//=======================================================================
//

void CoreInt::op_isTrue( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   token.exit( iself->asInteger() != 0 );
}

void CoreInt::op_toString( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   String s;
   token.exit( s.N(iself->asInteger()) );
}

void CoreInt::op_add( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

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

void CoreInt::op_sub( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

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


void CoreInt::op_mul( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

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


void CoreInt::op_div( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

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


void CoreInt::op_mod( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

   if( self->type() == op2->type() )
   {

      token.exit( self->asInteger() % op2->asInteger() );

   }
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
    
}


void CoreInt::op_pow( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   OpToken token( vm, self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_NUM )
   {

      token.exit( pow( self->forceNumeric(), op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    
}


void CoreInt::op_aadd( VMachine *vm, void*) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

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
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}

void CoreInt::op_asub( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

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
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreInt::op_amul( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

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
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreInt::op_adiv( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

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
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreInt::op_amod( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

   if( self->type() == op2->type() ) 
   {

      self->setInteger( self->asInteger() % op2->asInteger() );

   }
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreInt::op_apow( VMachine *vm, void* ) const {
    
   Item *self, *op2;

   vm->operands( self, op2 );

   if( self->type() == op2->type() || op2->type() == FLC_ITEM_NUM ) 
   {

      self->setNumeric( pow( self->forceNumeric() , op2->forceNumeric() ) );

   } 
   else 
      throw new OperandError( ErrorParam( e_invalid_op, __LINE__, __FILE__ ).origin( ErrorParam::e_orig_vm ).extra( "Invalid operand term" ) );
    

   vm->currentContext()->popData(); // Put self on the top of the stack
    
}


void CoreInt::op_inc(VMachine *vm, void* ) const {
    
   Item *self;

   vm->operands( self );

   self->setInteger( self->asInteger() + 1 );

}


void CoreInt::op_dec(VMachine *vm, void*) const {
    
   Item *self;

   vm->operands( self );

   self->setInteger( self->asInteger() + 1 );
    
}


void CoreInt::op_incpost(VMachine *, void* ) const {
    
   // TODO
    
}


void CoreInt::op_decpost(VMachine *, void* ) const {
    
   // TODO
    
}

}

/* end of coreint.cpp */
