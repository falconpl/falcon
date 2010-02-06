/*
   FALCON - The Falcon Programming Language.
   FILE: vm_run.cpp

   Implementation of virtual machine - main loop
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-09-08

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Virtual machine main loop.

   The main loop of the virtual machine deserves a space on its own.
*/

#include <falcon/vm.h>
#include <falcon/pcodes.h>
#include <falcon/vmcontext.h>
#include <falcon/sys.h>
#include <falcon/core_ext.h>
#include <falcon/sequence.h>

#include <falcon/coreobject.h>
#include <falcon/lineardict.h>
#include <falcon/string.h>
#include <falcon/cclass.h>
#include <falcon/carray.h>
#include <falcon/coredict.h>
#include <falcon/corefunc.h>
#include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/membuf.h>
#include <falcon/vmmsg.h>
#include <falcon/vmevent.h>
#include <falcon/rangeseq.h>
#include <falcon/generatorseq.h>
#include <falcon/garbagepointer.h>

#include <math.h>
#include <errno.h>
#include <string.h>

namespace Falcon {


Item *VMachine::getOpcodeParam( register uint32 bc_pos )
{
   Item *ret;

   // Load the operator ?
   switch( m_currentContext->code()[ m_currentContext->pc() + bc_pos ]  )
   {
      case P_PARAM_INT32:
         m_imm[bc_pos].setInteger( *reinterpret_cast<int32 *>( m_currentContext->code() + m_currentContext->pc_next() ) );
         m_currentContext->pc_next() += sizeof( int32 );
      return m_imm + bc_pos;

      case P_PARAM_INT64:
         m_imm[bc_pos].setInteger( loadInt64( m_currentContext->code() + m_currentContext->pc_next()  ) );
         m_currentContext->pc_next() += sizeof( int64 );
      return m_imm + bc_pos;

      case P_PARAM_STRID:
         {
            String *temp = currentLiveModule()->getString( *reinterpret_cast<int32 *>( m_currentContext->code() + m_currentContext->pc_next() ) );
            //m_imm[bc_pos].setString( temp, const_cast<LiveModule*>(currentLiveModule()) );
            m_imm[bc_pos].setString( temp );
            m_currentContext->pc_next() += sizeof( int32 );
         }
      return m_imm + bc_pos;

      case P_PARAM_LBIND:
         m_imm[bc_pos].setLBind( currentLiveModule()->getString(
            *reinterpret_cast<int32 *>( m_currentContext->code() + m_currentContext->pc_next() ) ) );
         m_currentContext->pc_next() += sizeof( int32 );
      return m_imm + bc_pos;

      case P_PARAM_NUM:
         m_imm[bc_pos].setNumeric( loadNum( m_currentContext->code() + m_currentContext->pc_next() ) );
         m_currentContext->pc_next() += sizeof( numeric );
      return m_imm + bc_pos;

      case P_PARAM_NIL:
         m_imm[bc_pos].setNil();
      return m_imm + bc_pos;

      case P_PARAM_UNB:
         m_imm[bc_pos].setUnbound();
      return m_imm + bc_pos;

      case P_PARAM_GLOBID:
      {
         register int32 id = *reinterpret_cast< int32 * >( m_currentContext->code() + m_currentContext->pc_next() );
         m_currentContext->pc_next()+=sizeof( int32 );
         return &moduleItem( id );
      }
      break;

      case P_PARAM_LOCID:
         ret = local( *reinterpret_cast< int32 * >( m_currentContext->code() + m_currentContext->pc_next() ) );
         m_currentContext->pc_next()+=sizeof(int32);
      return ret;

      case P_PARAM_PARID:
         ret = param( *reinterpret_cast< int32 * >( m_currentContext->code() + m_currentContext->pc_next() ) );
         m_currentContext->pc_next()+=sizeof(int32);
      return ret;

      case P_PARAM_TRUE:
         m_imm[bc_pos].setBoolean( true );
      return m_imm + bc_pos;

      case P_PARAM_FALSE:
         m_imm[bc_pos].setBoolean( false );
      return m_imm + bc_pos;

      case P_PARAM_NTD32: m_currentContext->pc_next() += sizeof(int32); return 0;
      case P_PARAM_NTD64: m_currentContext->pc_next() += sizeof(int64); return 0;
      case P_PARAM_REGA: return &regA();
      case P_PARAM_REGB: return &regB();
      case P_PARAM_REGS1: return &self();
      case P_PARAM_REGL1: return &latch();
      case P_PARAM_REGL2: return &latcher();
   }

   // we should not be here.
   fassert( false );
   return 0;
}


void VMachine::run()
{
   tOpcodeHandler *ops = m_opHandlers;

   // declare this as the running machine
   setCurrent();

   while( ! m_break )
   {
      try {
         // move m_currentContext->pc_next() to the end of instruction and beginning of parameters.
         // in case of opcode in the call_request range, this will move next pc to return.
         m_currentContext->pc_next() = m_currentContext->pc() + sizeof( uint32 );

         // external call required?
         if ( m_currentContext->pc() >= i_pc_call_request )
         {

            switch( m_currentContext->pc() )
            {
               case i_pc_call_external_ctor_return:
                  regA() = self();
                  //fallthrough
               case i_pc_call_external_return:
                  callReturn();
               break;

               // the request was just to ignore opcode
               case i_pc_redo_request:
                  if ( m_opCount )
                     m_opCount--; // prevent hitting oplimit
               break;

               default:
                  regA().setNil();
                  currentSymbol()->getExtFuncDef()->call( this );
            }
         }
         else
         {
            ops[ m_currentContext->code()[ m_currentContext->pc() ] ]( this );
         }

         m_opCount ++;

         //=========================
         // Executes periodic checks
         //

         if ( m_opCount >= m_opNextCheck )
         {
            // By default, if nothing else happens, we should do a check no sooner than this.
            m_opNextCheck += FALCON_VM_DFAULT_CHECK_LOOPS;

            // manage periodic callbacks
            periodicChecks();
         }

         // Jump to next isntruction.
         m_currentContext->pc() = m_currentContext->pc_next();
      }
      // catches explicitly raised items.
      catch( Item& raised )
      {
         handleRaisedItem( raised );
      }
      // errors thrown by C extensions;
      // they may get encapsulated in Error() instances and handled by the script
      // via handleRaised
      catch( Error *err )
      {
         handleRaisedError( err );
      }
   } // end while -- VM LOOP

   m_break = false;
}

/****************************************************
   Op code handlers
*****************************************************/

// 0
void opcodeHandler_END( register VMachine *vm )
{
   vm->regA().setNil();
   vm->terminateCurrentContext();
}


// 1
void opcodeHandler_NOP( register VMachine *vm )
{
}

// 2
void opcodeHandler_PSHN( register VMachine *vm )
{
   vm->stack().append( Item() );
}

// 3
void opcodeHandler_RET( register VMachine *vm )
{
   vm->retnil();
   vm->callReturn();
}

// 4
void opcodeHandler_RETA( register VMachine *vm )
{
   vm->callReturn();

   //? Check this -- I don't think is anymore necessary
   if( vm->regA().type() == FLC_ITEM_REFERENCE )
      vm->regA().copy( vm->regA().asReference()->origin() );
}

// 5
void opcodeHandler_PTRY( register VMachine *vm )
{
   register int32 target = vm->getNextNTD32();
   while( target > 0 )
   {
      vm->popTry( false );
      --target;
   }
}


// 6
void opcodeHandler_LNIL( register VMachine *vm )
{
   register Item *op1 = vm->getOpcodeParam( 1 )->dereference();
   op1->setNil();
}

// 7
void opcodeHandler_RETV(register VMachine *vm)
{
   vm->regA() = *vm->getOpcodeParam( 1 )->dereference();
   vm->callReturn();
}

// 8
void opcodeHandler_BOOL( register VMachine *vm )
{
   vm->regA().setBoolean( vm->getOpcodeParam( 1 )->dereference()->isTrue() );
}

// 9
void opcodeHandler_JMP( register VMachine *vm )
{
   vm->m_currentContext->pc_next() = vm->getNextNTD32();
}

// 0A
void opcodeHandler_GENA( register VMachine *vm )
{
   register uint32 size = (uint32) vm->getNextNTD32();
   CoreArray *array = new CoreArray( size );
   vm->regA().setArray( array );

   // copy the m-topmost items in the stack into the array
   if( size > 0 )
   {
      array->items().copyOnto( vm->stack(), vm->stack().length() - size,  size );
      vm->currentFrame()->pop( size );
   }
}

// 0B
void opcodeHandler_GEND( register VMachine *vm )
{
   register uint32 length = (uint32) vm->getNextNTD32();
   LinearDict *dict = new LinearDict( length );

   // copy the m-topmost items in the stack into the array
   uint32 len =  vm->stack().length();
   uint32 base = len - ( length * 2 );
   for ( uint32 i = base ; i < len; i += 2 ) {
      // insert may modify the stack (if using special "compare" functions)
      Item i1 = vm->stackItem(i);
      Item i2 = vm->stackItem(i+1);
      dict->put( i1, i2 );
      fassert( vm->stack().length() == len );
   }
   vm->stack().resize( base );
   vm->regA().setDict( new CoreDict(dict) );
}


// 0C
void opcodeHandler_PUSH( register VMachine *vm )
{
   /** \TODO Raise a stack overflow error on VM stack boundary limit. */
   Item *data = vm->getOpcodeParam( 1 )->dereference();
   if ( data->isFutureBind() )
   {
      // with this marker, the next call operation will search its parameters.
      // Let's consider this a temporary (but legitimate) hack.
      vm->regBind().flags( 0xF0 );
   }
   vm->stack().append( *data );
}

// 0D
void opcodeHandler_PSHR( register VMachine *vm )
{
   Item *referenced = vm->getOpcodeParam( 1 );
   if ( ! referenced->isReference() )
   {
      GarbageItem *ref = new GarbageItem( *referenced );
      referenced->setReference( ref );
   }

   vm->stack().append( *referenced );
}


// 0E
void opcodeHandler_POP( register VMachine *vm )
{
   if ( vm->stack().length() == 0 ) {
      vm->raiseHardError( e_stackuf, "POP", __LINE__ );
      return;
   }
   //  --- WARNING: do not dereference!
   vm->getOpcodeParam( 1 )->copy( vm->stack().back() );
   vm->stack().resize( vm->stack().length() - 1);
}


// 0F
void opcodeHandler_INC( register VMachine *vm )
{
   Item *operand =  vm->getOpcodeParam( 1 );
   operand->inc(vm->regA());
}

// 10
void opcodeHandler_DEC( register VMachine *vm )
{
   Item *operand =  vm->getOpcodeParam( 1 );
   operand->dec(vm->regA());
}


// 11
void opcodeHandler_NEG( register VMachine *vm )
{
   Item *operand = vm->getOpcodeParam( 1 );
   operand->neg( vm->regA() );
}


// 12
void opcodeHandler_NOT( register VMachine *vm )
{
   vm->regA().setInteger( vm->getOpcodeParam( 1 )->dereference()->isTrue() ? 0 : 1 );
}


//13
void opcodeHandler_TRAL( register VMachine *vm )
{
   if ( vm->stack().length() < 1 ) {
      vm->raiseHardError( e_stackuf, "TRAL", __LINE__ );
   }

   uint32 loop_last = vm->getNextNTD32();

   const Item &top = vm->stack().back();
   fassert( top.isGCPointer() );

   Iterator* iter = dyncast<Iterator*>(top.asGCPointer());
   // is this the last element?
   if (  ! iter->hasNext() )
   {
      if( iter->hasCurrent() )
         iter->next(); // position past last element

      vm->jump( loop_last );
   }
}



//14
void opcodeHandler_IPOP( register VMachine *vm )
{
   register uint32 amount = (uint32) vm->getNextNTD32();
   if ( vm->stack().length() < amount ) {
      vm->raiseHardError( e_stackuf, "IPOP", __LINE__ );
      return;
   }

   vm->stack().resize( vm->stack().length() - amount );
}

//15
void opcodeHandler_XPOP( register VMachine *vm )
{
   Item *operand = vm->getOpcodeParam( 1 )->dereference();
   // use copy constructor.
   Item itm( *operand );
   operand->copy( vm->stack()[vm->stack().length() - 1] );
   vm->stack()[vm->stack().length() - 1].copy( itm );
}

//16
void opcodeHandler_GEOR( register VMachine *vm )
{
   vm->regA().setRange(
      new CoreRange( (int32) vm->getOpcodeParam( 1 )->dereference()->forceIntegerEx() ) );
}

//17
void opcodeHandler_TRY( register VMachine *vm )
{
   vm->pushTry( vm->getNextNTD32() );
}

//18
void opcodeHandler_JTRY( register VMachine *vm )
{
   register int32 target = vm->getNextNTD32();

   vm->popTry( false );  // underflows are checked here
   vm->m_currentContext->pc_next() = target;
}

//19
void opcodeHandler_RIS( register VMachine *vm )
{
   // todo - when this will be in the main loop,
   // we may use directly handleRaisedEvent()
   throw *vm->getOpcodeParam( 1 )->dereference();
}

//1A
void opcodeHandler_BNOT( register VMachine *vm )
{
   register Item *operand = vm->getOpcodeParam( 1 )->dereference();
   if ( operand->type() == FLC_ITEM_INT ) {
      vm->regA().setInteger( ~operand->asInteger() );
   }
   else
      throw new TypeError( ErrorParam( e_bitwise_op ).extra("BNOT").origin( e_orig_vm ) );
}

//1B
void opcodeHandler_NOTS( register VMachine *vm )
{
   register Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( operand1->isOrdinal() )
      operand1->setInteger( ~operand1->forceInteger() );
   else
      throw new TypeError( ErrorParam( e_bitwise_op ).extra("NOTS").origin( e_orig_vm ) );
}

//1c
void opcodeHandler_PEEK( register VMachine *vm )
{
   register Item *operand = vm->getOpcodeParam( 1 )->dereference();

   if ( vm->stack().length() == 0 ) {
      vm->raiseHardError( e_stackuf, "PEEK", __LINE__ );
      return;
   }
   *operand = vm->stack().back();
}

// 1D
void opcodeHandler_FORK( register VMachine *vm )
{
   uint32 pSize = (uint32) vm->getNextNTD32();
   uint32 pJump = (uint32) vm->getNextNTD32();

   // create the coroutine
   vm->putAtSleep( vm->coPrepare( pSize ) );

   // fork
   vm->m_currentContext->pc_next() = pJump;
}

// 1D - Missing

// 1E
void opcodeHandler_LD( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   /*
   if( operand1->isLBind() && operand1->asLBind()->getCharAt(0) != '.' )
   {
      vm->bindItem( *operand1->asLBind(), *operand2 );
      vm->regA() =  *operand2;
   }
   else
   {
      switch( operand2->type() )
      {
         case FLC_ITEM_STRING:
            operand1->setString( new CoreString( *operand2->asString() ) );
            operand1->flags( operand2->flags() );
            break;

         case FLC_ITEM_LBIND:
            vm->unbindItem( *operand2->asLBind(), *operand1 );
            break;

         default:
            operand1->copy( *operand2 );
      }

      vm->regA() =  *operand1;
   }
   */

   if ( operand2->isString() )
   {
      operand1->setString( new CoreString( *operand2->asString() ) );
      operand1->flags( operand2->flags() );
   }
   else
      operand1->copy( *operand2 );

   vm->regA() =  *operand1;
}

// 1F
void opcodeHandler_LDRF( register VMachine *vm )
{
   // don't dereference
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );

   if( operand2 == &vm->m_imm[2] )
      operand1->setNil(); // breaks the reference
   else {
      // we don't use memPool::referenceItem for performace reasons
      GarbageItem *gitem;
      if ( operand2->isReference() )
      {
         gitem = operand2->asReference();
      }
      else
      {
         gitem = new GarbageItem( *operand2 );
         operand2->setReference( gitem );
      }

      operand1->setReference( gitem );
   }
}

// 20
void opcodeHandler_ADD( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 );
   Item *operand2 = vm->getOpcodeParam( 2 );

   Item target; // neutralize auto-ops
   operand1->add( *operand2, target );
   vm->regA() = target;
}

// 21
void opcodeHandler_SUB( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );

   Item target; // neutralize auto-ops
   operand1->sub( *operand2, target );
   vm->regA() = target;
}

// 22
void opcodeHandler_MUL( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   operand1->mul( *operand2, vm->regA() );
}

// 23
void opcodeHandler_DIV( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   operand1->div( *operand2, vm->regA() );
}


//24
void opcodeHandler_MOD( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   operand1->mod( *operand2, vm->regA() );
}

// 25
void opcodeHandler_POW( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   operand1->pow( *operand2, vm->regA() );
}

// 26
void opcodeHandler_ADDS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 );

   // TODO: S-operators
   operand1->add( *operand2, *operand1 );
   vm->regA() = *operand1;
}

//27
void opcodeHandler_SUBS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 );

   // TODO: S-operators
   operand1->sub( *operand2, *operand1 );
   vm->regA() = *operand1;
}


//28

void opcodeHandler_MULS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 );

   // TODO: S-operators
   operand1->mul( *operand2, *operand1 );
   vm->regA() = *operand1;
}

//29
void opcodeHandler_DIVS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 );

   // TODO: S-operators
   operand1->div( *operand2, *operand1 );
   vm->regA() = *operand1;
}


//2A
void opcodeHandler_MODS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 );

   // TODO: S-operators
   operand1->mod( *operand2, *operand1 );
   vm->regA() = *operand1;
}

//2B
void opcodeHandler_BAND( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      vm->regA().setInteger( operand1->forceInteger() & operand2->forceInteger() );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra("BAND").origin( e_orig_vm ) );
}

//2C
void opcodeHandler_BOR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      vm->regA().setInteger( operand1->forceInteger() | operand2->forceInteger() );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra("BOR").origin( e_orig_vm ) );
}

//2D
void opcodeHandler_BXOR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      vm->regA().setInteger( operand1->forceInteger() ^ operand2->forceInteger() );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra("BXOR").origin( e_orig_vm ) );
}

//2E
void opcodeHandler_ANDS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      operand1->setInteger( operand1->forceInteger() & operand2->forceInteger() );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra("ANDS").origin( e_orig_vm ) );
}

//2F
void opcodeHandler_ORS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->type() == FLC_ITEM_INT && operand2->type() == FLC_ITEM_INT ) {
      operand1->setInteger( operand1->asInteger() | operand2->asInteger() );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra("ORS").origin( e_orig_vm ) );
}

//30
void opcodeHandler_XORS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      operand1->setInteger( operand1->forceInteger() ^ operand2->forceInteger() );
   }
   else
      throw new TypeError( ErrorParam( e_invop ).extra("XORS").origin( e_orig_vm ) );
}

//31
void opcodeHandler_GENR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();
   Item *operand3 =  vm->getOpcodeParam( 3 )->dereference();


   int64 secondOp = operand2->forceIntegerEx();

   int64 step;
   if ( operand3->isNil() )
   {
      step = vm->stack()[ vm->stack().length() - 1].forceIntegerEx();
      vm->stack().resize( vm->stack().length() - 1);
   }
   else {
      step = operand3->forceIntegerEx();
   }


   int64 firstOp = operand1->forceIntegerEx();

   if( operand2->isOob() && firstOp <= secondOp )
      secondOp++;

   vm->regA().setRange( new CoreRange(
      firstOp,
      secondOp,
      step   ) );

}

//32
void opcodeHandler_EQ( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->regA().setBoolean( *operand1 == *operand2 );
}

//33
void opcodeHandler_NEQ( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->regA().setBoolean( *operand1 != *operand2 );
}

//34
void opcodeHandler_GT( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->regA().setBoolean( *operand1 > *operand2 );
}

//35
void opcodeHandler_GE( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->regA().setBoolean( *operand1 >= *operand2 );
}

//36
void opcodeHandler_LT( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->regA().setBoolean( *operand1 < *operand2 );
}

//37
void opcodeHandler_LE( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->regA().setBoolean( *operand1 <= *operand2 );
}

//38
void opcodeHandler_IFT( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand2->isTrue() )
      vm->m_currentContext->pc_next() = pNext;
}

//39
void opcodeHandler_IFF( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( ! operand2->isTrue() )
      vm->m_currentContext->pc_next() = pNext;
}

//3A
void opcodeHandler_CALL( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   operand2->readyFrame( vm, pNext );
}

//3B
void opcodeHandler_INST( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand2->type() != FLC_ITEM_CLASS )
   {
      throw new TypeError( ErrorParam( e_invop ).extra("INST").origin( e_orig_vm ) );
   }

   Item method = operand2->asClass()->constructor();
   if ( ! method.isNil() )
   {
      // set self in the function method.
      method.methodize( vm->self() );
      method.readyFrame( vm, pNext );
   }
}

//3C
void opcodeHandler_ONCE( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   const Symbol *call = 0;

   if ( operand2->isFunction() )
   {
      call = operand2->asFunction()->symbol();
   }
   else if ( operand2->isMethod() && operand2->asMethodFunc()->isFunc() )
   {
      call = static_cast<CoreFunc*>(operand2->asMethodFunc())->symbol();
   }

   if ( call != 0 && call->isFunction() )
   {
      // we suppose we're in the same module as the function things we are...
      register uint32 itemId = call->getFuncDef()->onceItemId();
      if ( vm->moduleItem( itemId ).isNil() )
         vm->moduleItem( itemId ).setInteger( 1 );
      else
         vm->m_currentContext->pc_next() = pNext;
      return;
   }

   vm->raiseHardError( e_invop, "ONCE", __LINE__ );
}

//3D
void opcodeHandler_LDV( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->latch() = *operand1;
   vm->latcher() = *operand2;

   operand1->getIndex( *operand2, vm->regA() );
}

//3E
void opcodeHandler_LDP( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();
   vm->latch() = *operand1;
   vm->latcher() = *operand2;

   if( operand2->isString() )
   {
      String *property = operand2->asString();
      operand1->getProperty( *property, vm->regA() );
   }
   else
      throw
         new AccessError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
            extra( operand2->isString() ? *operand2->asString() : "?" ) );
}


// 3F
void opcodeHandler_TRAN( register VMachine *vm )
{
   if ( vm->stack().length() < 1 ) {
      vm->raiseHardError( e_stackuf, "TRAN", __LINE__ );
   }

   uint32 loop_begin = vm->getNextNTD32();
   uint32 varCount = vm->getNextNTD32();
   const Item &top = vm->stack().back();
   fassert( top.isGCPointer() );

   Iterator* iter = dyncast<Iterator*>(top.asGCPointer());
   if( iter->isValid() && iter->next() )
   {
      // Ok, we proceeded to the next item; prepare the vars
      vm->expandTRAV( varCount, *iter );
      vm->currentContext()->pc_next() = loop_begin;
   }
   else
   {
      vm->currentContext()->pc_next() += sizeof(uint32) * 2 * varCount;
   }
}

//40
void opcodeHandler_LDAS( register VMachine *vm )
{
   uint32 size = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand2->isArray() )
   {
      CoreArray* arr = operand2->asArray();
      if( arr->length() != size )
      {
         throw
            new AccessError( ErrorParam( e_unpack_size )
               .extra( "LDAS" )
               .origin( e_orig_vm ) );
      }
      else {
         uint32 oldpos = vm->stack().length();
         vm->stack().resize( oldpos + size );
         void* mem = &vm->stack()[ oldpos ];;
         memcpy( mem, arr->items().elements(), size * sizeof(Item) );
     }
   }
   else {
      throw
            new AccessError( ErrorParam( e_invop )
               .extra( "LDAS" )
               .origin( e_orig_vm ) );
   }
}

//41
void opcodeHandler_SWCH( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();
   uint64 sw_count = (uint64) vm->getNextNTD64();

   byte *tableBase =  vm->m_currentContext->code() + vm->m_currentContext->pc_next();

   uint16 sw_int = (int16) (sw_count >> 48);
   uint16 sw_rng = (int16) (sw_count >> 32);
   uint16 sw_str = (int16) (sw_count >> 16);
   uint16 sw_obj = (int16) sw_count;

   //determine the value type to be checked
   switch( operand2->type() )
   {
      case FLC_ITEM_NIL:
         if ( *reinterpret_cast<uint32 *>( tableBase ) != 0xFFFFFFFF ) {
            vm->m_currentContext->pc_next() = *reinterpret_cast<uint32 *>( tableBase );
            return;
         }
      break;

      case FLC_ITEM_INT:
         if ( sw_int > 0 &&
               vm->seekInteger( operand2->asInteger(), tableBase + sizeof(uint32), sw_int, vm->m_currentContext->pc_next() ) )
            return;
         if ( sw_rng > 0 &&
               vm->seekInRange( operand2->asInteger(),
                     tableBase + sizeof(uint32) + sw_int * (sizeof(uint64) + sizeof(uint32) ),
                     sw_rng, vm->m_currentContext->pc_next() ) )
            return;
      break;

      case FLC_ITEM_NUM:
         if ( sw_int > 0 &&
               vm->seekInteger( operand2->forceInteger(), tableBase + sizeof(uint32), sw_int, vm->m_currentContext->pc_next() ) )
            return;
         if ( sw_rng > 0 &&
               vm->seekInRange( operand2->forceInteger(),
                     tableBase + sizeof(uint32) + sw_int * (sizeof(uint64) + sizeof(uint32) ),
                     sw_rng, vm->m_currentContext->pc_next() ) )
            return;
      break;

      case FLC_ITEM_STRING:
         if ( sw_str > 0 &&
               vm->seekString( operand2->asString(),
                  tableBase + sizeof(uint32) +
                     sw_int * (sizeof(uint64) + sizeof(uint32) )+
                     sw_rng * (sizeof(uint64) + sizeof(uint32) ),
                     sw_str,
                     vm->m_currentContext->pc_next() ) )
            return;
      break;
   }

   // always tries with the objects, if given and had not success up to here.
   if ( sw_obj > 0 &&
         vm->seekItem( operand2,
            tableBase + sizeof(uint32) +
               sw_int * (sizeof(uint64) + sizeof(uint32) ) +
               sw_rng * (sizeof(uint64) + sizeof(uint32) ) +
               sw_str * (sizeof(uint32) + sizeof(uint32) ),
               sw_obj,
               vm->m_currentContext->pc_next() ) )
            return;

   // in case of failure...
   vm->m_currentContext->pc_next() = pNext;
}


//46
void opcodeHandler_IN( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   bool result = false;

   switch( operand2->type() )
   {
      case FLC_ITEM_STRING:
         if( operand1->type() == FLC_ITEM_STRING )
            result = operand2->asString()->find( *operand1->asString() ) != csh::npos;
      break;

      case FLC_ITEM_ARRAY:
      {
         Item *elements =  operand2->asArray()->items().elements();
         for( uint32 pos = 0; pos <operand2->asArray()->length(); pos++ )
         {
            if( elements[pos] == *operand1 ) {
               result = true;
               break;
            }
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         result = operand2->asDict()->find( *operand1 ) != 0;
      }
      break;

      case FLC_ITEM_OBJECT:
         if( operand1->type() == FLC_ITEM_STRING )
            result = operand2->asObjectSafe()->hasProperty( *operand1->asString() );
      break;

      case FLC_ITEM_CLASS:
      {
         if( operand1->type() == FLC_ITEM_STRING )
         {
            uint32 pos;
            result = operand2->asClass()->properties().findKey( *operand1->asString(), pos ) != 0;
         }
      }
      break;
   }

   vm->regA().setBoolean( result );
}

//47
void opcodeHandler_NOIN( register VMachine *vm )
{
   // do not decode operands; IN will do it
   opcodeHandler_IN( vm );
   vm->regA().setBoolean( ! vm->regA().asBoolean() );
}

//48
void opcodeHandler_PROV( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand2->type() != FLC_ITEM_STRING ) {
      vm->raiseHardError( e_invop, "PROV", __LINE__  ); // hard error, as the string should be created by the compiler
      return;
   }

   bool result;
   uint32 pos;
   switch ( operand1->type() ) {
      case FLC_ITEM_CLASS:
         result = operand1->asClass()->properties().findKey( *operand2->asString(), pos ) != 0;
      break;

      case FLC_ITEM_OBJECT:
         result = operand1->asObjectSafe()->hasProperty( *operand2->asString() );
      break;

      case FLC_ITEM_ARRAY:
         result = operand1->asArray()->getProperty( *operand2->asString() )!=0;
      break;
      case FLC_ITEM_DICT:
         result = operand1->asDict()->isBlessed() &&
                  operand1->asDict()->find( *operand2->asString() )!=0;
      break;
      default:
         result = false;
   }

   vm->regA().setBoolean( result );
}



//49
void opcodeHandler_STVS( register VMachine *vm )
{
   if(  vm->stack().empty() )
   {
      vm->raiseHardError( e_stackuf, "STVS", __LINE__  );
      return;
   }

   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   Item *origin = &vm->stack()[vm->stack().length() - 1];

   operand1->setIndex( *operand2, *origin );
   vm->regA() = *origin;
   vm->stack().resize( vm->stack().length() - 1);

}

//4A
void opcodeHandler_STPS( register VMachine *vm )
{
   if( vm->stack().empty() )
   {
      vm->raiseHardError( e_stackuf, "STPS", __LINE__  );
      return;
   }

   Item *target =  vm->getOpcodeParam( 1 );
   Item *method = vm->getOpcodeParam( 2 )->dereference();

   if ( method->isString() )
   {
      target->setProperty( *method->asString(), vm->stack()[vm->stack().length() - 1] );
      vm->regA() = vm->stack()[vm->stack().length() - 1];
      vm->stack().resize( vm->stack().length() - 1 );
   }
   else
   {
      vm->stack().resize( vm->stack().length() - 1 );
      throw new TypeError( ErrorParam( e_prop_acc ).extra("STPS").origin( e_orig_vm ) );
   }
}

//4B
void opcodeHandler_AND( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->regA().setBoolean( operand1->isTrue() && operand2->isTrue() );
}

//4C
void opcodeHandler_OR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->regA().setBoolean( operand1->isTrue() || operand2->isTrue() );
}

//50
void opcodeHandler_STV( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 );
   Item *operand2 = vm->getOpcodeParam( 2 );
   Item *origItm = vm->getOpcodeParam( 3 );
   Item *origin = origItm->dereference();

   operand1->setIndex( *operand2, *origin );

   if( origItm != &vm->regB() )
         vm->regA() = *origin;
}

//51
void opcodeHandler_STP( register VMachine *vm )
{
   Item *target = vm->getOpcodeParam( 1 );
   Item *method = vm->getOpcodeParam( 2 )->dereference();
   Item *sourcend = vm->getOpcodeParam( 3 );
   Item *source = sourcend->dereference();


   if ( method->isString() )
   {
      target->setProperty( *method->asString(), *source );

      // when B is the source, the right value is already in A.
      if( sourcend != &vm->regB() )
         vm->regA() = *source;
      return;
   }

   throw new TypeError( ErrorParam( e_invop ).extra("STP").origin( e_orig_vm ) );

/**
   \todo check this code
   This code originally allowed to load a method of an object into a method
   of another object. This is considered now too danjerous, as external
   methods can rely on how the object is built. So, it's fine to set a
   function into a method, but it's not fine to extract a method or
   to exchange methods between classes.

   An alternative could be that to force methods to check for user
   data signature, but that was exactly what I wanted to avoid when
   putting user data in objects.

   Giancarlo Niccolai

   \code

            if( source->type() == FLC_ITEM_METHOD ) {
               // will instantiate by value
               if( target->asObject()->setProperty( vm->m_op2->asString(), source->asMethodFunction() ) )
                  return;
            }
            else if ( source->type() == FLC_ITEM_EXTMETHOD )
            {
               Item itm;
               itm.setExtFunc(source->asMethodExtFunc());
               if( target->asObject()->setProperty( vm->m_op2->asString(), itm ) )
                  return;
            }
            else {
               if( target->asObject()->setProperty( vm->m_op2->asString(), *source ) )
                  return;
            }
         }
   \endcode
*/
}

//52
void opcodeHandler_LDVT( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->latch() = *operand1;
   vm->latcher() = *operand2;

   operand1->getIndex( *operand2, *vm->getOpcodeParam( 3 ) );
}

//53
void opcodeHandler_LDPT( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();
   vm->latch() = *operand1;
   vm->latcher() = *operand2;

   if( operand2->isString() )
   {
      String *property = operand2->asString();
      operand1->getProperty( *property, *vm->getOpcodeParam( 3 ) );
   }
   else
      throw
         new AccessError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
            extra( operand2->isString() ? *operand2->asString() : "?" ) );
}

//54
void opcodeHandler_STVR( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 );
   Item *operand2 = vm->getOpcodeParam( 2 );

   // do not deref op3
   Item *origin = vm->getOpcodeParam( 3 );

   //STV counts as an assignment, we must copy to expression value.
   if( origin != &vm->regA())
      vm->regA() = *origin;

   GarbageItem *gitem;
   if( ! origin->isReference() )
   {
      gitem = new GarbageItem( *origin );
      origin->setReference( gitem );
   }

   operand1->setIndex( *operand2, *origin );
}

//55
void opcodeHandler_STPR( register VMachine *vm )
{
   Item *target = vm->getOpcodeParam( 1 );
   Item *method = vm->getOpcodeParam( 2 )->dereference();

   Item *source = vm->getOpcodeParam( 3 );
   vm->regA() = *source;

   if ( method->isString() )
   {
      target->setProperty( *method->asString(), *source );
      return;
   }

   throw new TypeError( ErrorParam( e_prop_acc ).extra("STPR").origin( e_orig_vm ) );
}

//56
void opcodeHandler_TRAV( register VMachine *vm )
{

   // get the jump label.
   uint32 wayout = (uint32) vm->getNextNTD32();
   // get the number of variables in the trav loop
   uint32 varcount = (uint32) vm->getNextNTD32();
   Item *source = vm->getOpcodeParam( 3 )->dereference();

   // our sequence:
   Sequence *seq = 0;

   switch( source->type() )
   {
   case FLC_ITEM_ARRAY:
      seq = &source->asArray()->items();
      break;

   case FLC_ITEM_DICT:
      seq = &source->asDict()->items();
      break;

   case FLC_ITEM_OBJECT:
      seq = source->asObjectSafe()->getSequence();
      // is the object a direct sequence?
      if( seq == 0 )
      {
         // is it a callable entity?
         if ( source->asObjectSafe()->hasProperty( OVERRIDE_OP_CALL ))
         {
            seq = new GeneratorSeq( vm, *source );
            GarbagePointer* ptr = new GarbagePointer( seq );
            seq->owner( ptr );
         }
         else
            // we have mess on the stack, but it doesn't matter here.
            throw new TypeError( ErrorParam( e_invop ).extra("TRAV").origin( e_orig_vm ) );
      }
      break;


   case FLC_ITEM_RANGE:
      // optimize: avoid creating a bit of stuff if not needed
      if( source->asRangeIsOpen() ||
          (source->asRangeEnd() == source->asRangeStart() && source->asRangeStep() == 0) ||
          ( source->asRangeStep() > 0 && source->asRangeStart() > source->asRangeEnd() ) ||
          ( source->asRangeStep() < 0 && source->asRangeStart() < source->asRangeEnd() )
         )
      {
         vm->jump( wayout );
         return;
      }

      // ok, the range can generate a sequence. Hence...
      try {
         seq = new RangeSeq( *source->asRange() );
         // also, save the sequence in a garbage sensible item.
         // we don't record anywhere this item, but it will be marked
         // as long as we have the iterator active in the stack.
         GarbagePointer* ptr = new GarbagePointer( seq );
         seq->owner( ptr );
         seq->gcMark( vm->generation() ); // will mark our pointer
      }
      catch( ... ) {
         fassert( false ); // must not throw
      }
      break;

   case FLC_ITEM_NIL:
      // jump out
      vm->jump( wayout );
      return;


   default:
      if( source->isCallable() )
      {
         seq = new GeneratorSeq( vm, *source );
         GarbagePointer* ptr = new GarbagePointer( seq );
         seq->owner( ptr );
      }
      else
      {
         throw new TypeError( ErrorParam( e_invop ).extra("TRAV").origin( e_orig_vm ) );
      }
   }

   // empty sequence?
   if ( seq == 0 || seq->empty() )
   {
      vm->jump(wayout);
      return;
   }

   // create the iterator and push it.
   Item iterItem;
   Iterator* current = new Iterator( seq );
   iterItem.setGCPointer( current );
   vm->pushParam( iterItem );

   // the source is safe through back-marking of the iterator on its sequence,
   // and of the sequence on its owner.

   // now distribute the current value of the element on the variables.
   vm->expandTRAV( varcount, *current );
}


//57
void opcodeHandler_INCP( register VMachine *vm )
{
   Item *operand =  vm->getOpcodeParam( 1 )->dereference();
   Item temp;
   operand->incpost( temp );

   vm->regB() = *operand;
   vm->regA() = temp;
}

//58
void opcodeHandler_DECP( register VMachine *vm )
{
   Item *operand =  vm->getOpcodeParam( 1 )->dereference();
   Item temp;
   operand->decpost( temp );

   vm->regB() = *operand;
   vm->regA() = temp;
}

//59
void opcodeHandler_SHL( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      vm->regA().setInteger( operand1->asInteger() << operand2->asInteger() );
   else
      throw
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHL") );

}

//5A
void opcodeHandler_SHR( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      vm->regA().setInteger( operand1->asInteger() >> operand2->asInteger() );
   else
      throw
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHR") );
}

//5B
void opcodeHandler_SHLS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      operand1->setInteger( operand1->asInteger() << operand2->asInteger() );
   else
      throw
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHLS") );
}

//5C
void opcodeHandler_SHRS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      operand1->setInteger( operand1->asInteger() >> operand2->asInteger() );
   else
      throw
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHRS") );
}

//5D
void opcodeHandler_CLOS( register VMachine *vm )
{
   uint32 size = (uint32) vm->getNextNTD32();
   Item *tgt = vm->getOpcodeParam( 2 )->dereference();
   Item *src = vm->getOpcodeParam( 3 )->dereference();

   fassert( src->isFunction() );
   *tgt = new CoreFunc( *src->asFunction() );

   if( size > 0 )
   {
      if ( size > vm->stack().length() )
      {
         throw
            new CodeError( ErrorParam( e_stackuf ).origin( e_orig_vm ).extra("CLOS") );
      }

      ItemArray* closure = new ItemArray( size );
      Item *data = closure->elements();
      int32 base = vm->stack().length() - size;
      memcpy( data, &vm->stack()[ base ], sizeof(Item)*size );
      closure->length( size );
      vm->stack().resize( base );

      tgt->asFunction()->closure( closure );
   }
}

// 5D
void opcodeHandler_PSHL( register VMachine *vm )
{
   vm->stack().append( *vm->getOpcodeParam( 1 ) );
}


// 5E
void opcodeHandler_POWS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 );

   // TODO: S-operators
   operand1->pow( *operand2, *operand1 );
   vm->regA() = *operand1;
}

//5F
void opcodeHandler_LSB( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_STRING << 8 | FLC_ITEM_INT:
      case FLC_ITEM_STRING << 8 | FLC_ITEM_NUM:
      {
         int32 pos = (int32) operand2->forceInteger();
         String *cs = operand1->asString();
         if ( cs->checkPosBound( pos ) )
            vm->retval( (int64) cs->getCharAt( pos ) );
         else
            throw
               new AccessError( ErrorParam( e_arracc ).origin( e_orig_vm ).extra( "LSB" ) );
      }
      return;
   }

   throw
      new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra( "LSB" ) );
}

// 60
void opcodeHandler_EVAL( register VMachine *vm )
{
   // We know the first operand must be a string
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();

   if ( operand1->isArray() )
   {
      CoreArray *arr = operand1->asArray();
      if ( arr->length() > 0 ) {
         // fake as if we were called by a function
         // This will cause functionalEval to produce a correct return frame
         // in case it needs sub functional evals.
         vm->createFrame(0);
         uint32 savePC = vm->m_currentContext->pc_next();
         vm->m_currentContext->pc_next() = VMachine::i_pc_call_external_return;
         if( ! vm->functionalEval( *operand1 ) )
         {
            // it wasn't necessary; reset pc to the correct value
            vm->callReturn();
            vm->m_currentContext->pc_next() = savePC;
         }
         // ok here we're ready either to jump where required by functionalEval or
         // to go on as usual
         return;
      }
   }
   else if ( operand1->isCallable() ) {
      vm->callFrame( *operand1, 0 );
      return;
   }

   // by default, just copy the operand
   vm->regA() = *operand1;
}

//61
void opcodeHandler_SELE( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *real_op2 = vm->getOpcodeParam( 2 );
   Item *op2 = real_op2->dereference();
   uint64 sw_count = (uint64) vm->getNextNTD64();

   byte *tableBase = vm->m_currentContext->code() + vm->m_currentContext->pc_next();

   // SELE B has a special semantic used as a TRY/CATCH helper
   bool hasDefault = *reinterpret_cast<uint32 *>( tableBase )  == 0;
   bool reRaise = real_op2 == &vm->regB();

   // In select we use only integers and object fields.
   uint16 sw_int = (int16) (sw_count >> 48);
   uint16 sw_obj = (int16) sw_count;

   //determine the value type to be checked
   int type = op2->type();
   if ( sw_int > 0 )
   {
      if ( type == FLC_ITEM_INT )
         type = FLC_ITEM_NUM;

      if ( vm->seekInteger( type, tableBase + sizeof(uint32), sw_int, vm->m_currentContext->pc_next() ) )
         return;
   }

   if (sw_obj > 0 )
   {
      // we have to search the symbol of the class of our object
      // always tries with the objects, if given and had not success up to here.
      if ( vm->seekItemClass( op2,
               tableBase + sizeof(uint32) +
                  sw_int * (sizeof(uint64) + sizeof(uint32) ),
                  sw_obj,
                  vm->m_currentContext->pc_next() ) )
               return;
   }

   if ( reRaise && hasDefault ) // in case of re-raise
   {
      throw vm->regB();
   }

   // in case of failure go to default or past sele...
   vm->m_currentContext->pc_next() = pNext;
}

//62
void opcodeHandler_INDI( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( ! operand1->isString() )
   {
      throw new TypeError( ErrorParam( e_invop ).extra("INDI").origin( e_orig_vm ) );
   }

   if ( ! vm->findLocalVariable( *operand1->asString(), vm->regA() ) )
   {
       throw
            new ParamError( ErrorParam( e_param_indir_code ).origin( e_orig_vm ).extra( "INDI" ) );
   }
}

//63
void opcodeHandler_STEX( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( ! operand1->isString() )
   {
      throw new TypeError( ErrorParam( e_invop ).extra("STEX").origin( e_orig_vm ) );
   }

   CoreString *target = new CoreString;
   String *src = operand1->asString();

   VMachine::returnCode retval = vm->expandString( *src, *target );
   switch( retval )
   {
      case VMachine::return_ok:
         vm->regA() = target;
      break;

      case VMachine::return_error_parse_fmt:
         throw
            new TypeError( ErrorParam( e_fmt_convert ).origin( e_orig_vm ).extra( "STEX-format" ) );
      break;

      case VMachine::return_error_string:
         throw
            new ParamError( ErrorParam( e_param_strexp_code ).origin( e_orig_vm ).extra( "STEX-string" ) );
      break;

      case VMachine::return_error_parse:
         throw
               new ParamError( ErrorParam( e_param_indir_code ).origin( e_orig_vm ).extra( "STEX-parse" ) );
      break;

      default:  // warning no-op
         return;
   }
}

//64
void opcodeHandler_TRAC( register VMachine *vm )
{
   if ( vm->stack().length() < 1 ) {
      vm->raiseHardError( e_stackuf, "TRAC", __LINE__ );
   }

   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   const Item &top = vm->stack().back();
   fassert( top.isGCPointer() );
   Iterator* iter = dyncast<Iterator*>(top.asGCPointer());

   if( operand1->isString() )
      iter->getCurrent() = new CoreString( *operand1->asString() );
   else
      iter->getCurrent() = *operand1;
}

//65
void opcodeHandler_WRT( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   Stream *stream = vm->stdOut();
   if ( stream == 0 )
   {
      vm->raiseHardError( e_invop, "WRT", __LINE__  );
      return;
   }

   if( operand1->isString() )
   {
      String *s = operand1->asString();
      stream->writeString( *s );
   }
   else {
      String temp;
      vm->itemToString( temp, operand1 );
      stream->writeString( temp );
   }

   stream->flush();
}


void opcodeHandler_STO( register VMachine *vm )
{
   // STO is like LD, but it doesn't dereference param 1.
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   /*if ( operand2->isString() )
      operand1->setString( new CoreString( *operand2->asString() ) );
   else*/
      operand1->copy( *operand2 );

   vm->regA() = *operand1;
}

// 0x67
void opcodeHandler_FORB( register VMachine *vm )
{
   // We know the first operand must be a string
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( ! operand1->isString() )
   {
      vm->raiseHardError( e_invop, "FORB", __LINE__  );
      return;
   }

   vm->regA().setLBind( new CoreString( *operand1->asString() ),
      new GarbageItem( *operand2 ) );
}

// 0x68
void opcodeHandler_OOB( register VMachine *vm )
{
   uint32 pmode = (uint32) vm->getNextNTD32();
   Item *operand =  vm->getOpcodeParam( 2 )->dereference();
   switch( pmode )
   {
      case 0:
         vm->regA() = *operand;
         vm->regA().setOob( false );
         break;

      case 1:
         vm->regA() = *operand;
         vm->regA().setOob( true );
         break;

      case 2:
         vm->regA() = *operand;
         vm->regA().setOob( ! operand->isOob() );
         break;

      default:
         vm->regA().setBoolean( operand->isOob() );
   }
}

// 0x69
void opcodeHandler_TRDN( register VMachine *vm )
{
   if ( vm->stack().length() < 1 ) {
      vm->raiseHardError( e_stackuf, "TRAC", __LINE__ );
   }

   uint32 loop_begin = vm->getNextNTD32();
   uint32 loop_out = vm->getNextNTD32();
   uint32 vars = vm->getNextNTD32();

   const Item &top = vm->stack().back();
   fassert( top.isGCPointer() );
   Iterator* iter = dyncast<Iterator*>(top.asGCPointer());

   if( vars == 0 )
   {
      fassert( iter->hasPrev() );
      iter->prev();
      iter->erase();
      vm->jump( loop_begin ); // actually, the "out" pointer
   }
   else
   {
      // a normal TRDN
      iter->erase();
      if( iter->hasCurrent() )
      {
         vm->expandTRAV( vars, *iter );
         vm->jump( loop_begin );
      }
      else
         vm->jump( loop_out );
   }
}

// 0x70
void opcodeHandler_EXEQ( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   vm->regA().setBoolean( operand1->exactlyEqual(*operand2) );
}
   

}

/* end of vm_run.cpp */
