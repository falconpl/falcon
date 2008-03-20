/*
   FALCON - The Falcon Programming Language.
   FILE: vm_run.cpp
   $Id: vm_run.cpp,v 1.60 2007/08/18 12:07:37 jonnymind Exp $

   Implementation of virtual machine - main loop
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-09-08
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

#include <falcon/cobject.h>
#include <falcon/lineardict.h>
#include <falcon/string.h>
#include <falcon/cclass.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/attribute.h>
#include <falcon/membuf.h>

#include <math.h>
#include <errno.h>

namespace Falcon {


Item *VMachine::getOpcodeParam( register uint32 bc_pos )
{
   Item *ret;

   // Load the operator ?
   switch( m_code[ m_pc + bc_pos ]  )
   {
      case P_PARAM_INT32:
         m_imm[bc_pos].setInteger( endianInt32(*reinterpret_cast<int32 *>( m_code + m_pc_next ) ) );
         m_pc_next += sizeof( int32 );
      return m_imm + bc_pos;

      case P_PARAM_INT64:
         m_imm[bc_pos].setInteger( endianInt64(*reinterpret_cast<int64 *>( m_code + m_pc_next ) ) );
         m_pc_next += sizeof( int64 );
      return m_imm + bc_pos;

      case P_PARAM_STRID:
		 m_imm[bc_pos].setString( const_cast< String *>( m_currentModule->getString( endianInt32(*reinterpret_cast<int32 *>( m_code + m_pc_next ) ) ) ) );
		 m_pc_next += sizeof( int32 );
      return m_imm + bc_pos;

      case P_PARAM_NUM:
         m_imm[bc_pos].setNumeric( endianNum( *reinterpret_cast<numeric *>( m_code + m_pc_next ) ) );
         m_pc_next += sizeof( numeric );
      return m_imm + bc_pos;

      case P_PARAM_NIL:
         m_imm[bc_pos].setNil();
      return m_imm + bc_pos;

      case P_PARAM_GLOBID:
      {
         register int32 id = endianInt32(*reinterpret_cast< int32 * >( m_code + m_pc_next ) );
         m_pc_next+=sizeof( int32 );
         return &moduleItem( id );
      }
      break;

      case P_PARAM_LOCID:
         ret = &stackItem( m_stackBase +
               endianInt32(*reinterpret_cast< int32 * >( m_code + m_pc_next ) ) );
         m_pc_next+=sizeof(int32);
      return ret;

      case P_PARAM_PARID:
         ret = &stackItem( m_stackBase - paramCount() - VM_FRAME_SPACE +
               endianInt32(*reinterpret_cast< int32 * >( m_code + m_pc_next ) ) );
         m_pc_next+=sizeof(int32);
      return ret;

      case P_PARAM_TRUE:
         m_imm[bc_pos].setBoolean( true );
      return m_imm + bc_pos;

      case P_PARAM_FALSE:
         m_imm[bc_pos].setBoolean( false );
      return m_imm + bc_pos;

      case P_PARAM_NTD32: m_pc_next += sizeof(int32); return 0;
      case P_PARAM_NTD64: m_pc_next += sizeof(int64); return 0;
      case P_PARAM_REGA: return &m_regA;
      case P_PARAM_REGB: return &m_regB;
      case P_PARAM_REGS1: return &m_regS1;
      case P_PARAM_REGS2: return &m_regS2;
   }

   // we should not be here.
   fassert( false );
   return 0;
}


void VMachine::run()
{
   tOpcodeHandler *ops = m_opHandlers;
   m_event = eventNone;

   while( 1 )
   {
      // move m_pc_next to the end of instruction and beginning of parameters.
      // in case of opcode in the call_request range, this will move next pc to return.
      m_pc_next = m_pc + sizeof( uint32 );

      // external call required?
      if ( m_pc >= i_pc_call_request )
      {
         switch( m_pc )
         {
            case i_pc_call_external_ctor_return:
               m_regA = m_regS1;
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
               m_symbol->getExtFuncDef()->call( this );
         }
      }
      else
      {
         // execute the opcode.
         ops[ m_code[ m_pc ] ]( this );
      }

      m_opCount ++;

      //=========================
      // Executes periodic checks
      if ( m_opCount > m_opNextCheck )
      {
         if ( m_opCount > m_opNextGC )
         {
            m_memPool->checkForGarbage();
            m_opNextGC = m_opCount + m_loopsGC;
            m_opNextCheck = m_opNextGC;
         }

         if ( m_opLimit > 0 )
         {
            // Bail out???
            if ( m_opCount > m_opLimit )
               return;
            else
               if ( m_opNextCheck > m_opLimit )
                  m_opNextCheck = m_opLimit;
         }

         if( ! m_atomicMode )
         {
            if( m_allowYield && ! m_sleepingContexts.empty() && m_opCount > m_opNextContext ) {
               rotateContext();
               m_opNextContext = m_opCount + m_loopsContext;
               if( m_opNextContext < m_opNextCheck )
                  m_opNextCheck = m_opNextContext;
            }

            // Periodic Callback
            if( m_loopsCallback > 0 && m_opCount > m_opNextCallback )
            {
               periodicCallback();
               m_opNextCallback = m_opCount + m_loopsCallback;
               if( m_opNextCallback < m_opNextCheck )
                  m_opNextCheck = m_opNextCallback;
            }

            // in case of single step:
            if( m_bSingleStep )
            {
               // stop also next op
               m_opNextCheck = m_opCount + 1;
               return; // maintain the event we have, but exit now.
            }
         }
      }

      //===============================
      // consider requests.
      //
      switch( m_event )
      {

         // This events leads to VM main loop exit.
         case eventInterrupt:
         case eventSuspend:
            if ( m_atomicMode )
            {
               raiseError( new InterruptedError( ErrorParam( e_interrupted ).origin( e_orig_vm ).
                     symbol( "itemToString" ).
                     module( "core.vm" ).
                     line( __LINE__ ).
                     hard() ) );
               // just re-parse the event
               m_pc = i_pc_redo_request;
               continue;
            }

            m_pc = m_pc_next;
         return;

         // event return is used to return from a temporary frame,
         // so it is better to reset it on exit so that the calling function
         // do not need this burden.
         case eventReturn:
            m_pc = m_pc_next;
            resetEvent();
         return;

         // manage try/catch
         case eventRisen:
            // While the try frame is not in the current frame, we should return.
            // this unless m_stackBase is zero; in that case, the VM must take some action.

            // However, before proceding we have to create a correct stack frame report for the error.
            // If the error is internally generated, a frame has been already created. We should
            // create here only error data about uncaught raises from the scripts.

            if( m_tryFrame == i_noTryFrame && m_error == 0 )  // uncaught error raised from scripts...
            {
               // create the error that the external application will see.
               Error *err;
               if ( m_regB.isOfClass( "Error" ) )
               {
                  // in case of an error of class Error, we have already a good error inside of it.
                  ErrorCarrier *car = (ErrorCarrier *)m_regB.asObject()->getUserData();
                  err = car->error();
                  err->incref();
               }
               else {
                  // else incapsulate the item in an error.
                  err = new GenericError( ErrorParam( e_uncaught ).origin( e_orig_vm ) );
                  err->raised( m_regB );
               }

               fillErrorContext( err );

               m_error = err;
            }

            // Enter the stack frame that should handle the error (or raise to the top if uncaught)
            while( m_stackBase != 0 && ( m_stackBase > m_tryFrame || m_tryFrame == i_noTryFrame ) )
            {
               callReturn();
               if ( m_event == eventReturn )
               {
                  // yes, exit but of course, maintain the error status
                  m_event = eventRisen;

				  // we must return only if the stackbase is not zero; otherwise, we return to a
				  // base callItem, and we must manage internally that case.
				  if ( m_stackBase != 0 )
                      return;
               }
               // call return may raise eventQuit, but only when m_stackBase is zero,
               // so we don't consider it.
            }

            // We are in the frame that should handle the error, in one way or another
            // should we catch it?
            // If the error is zero, we know we have a script exception ready to be caught
            // as we have filtered it before
            if ( m_error == 0 )
            {
               popTry( true );
               m_event = eventNone;
               continue;
            }
            // else catch it only if allowed.
            else if( m_error->catchable() && m_tryFrame != i_noTryFrame )
            {
               CoreObject *obj = m_error->scriptize( this );

               if ( obj != 0 )
               {
                  // we'll manage the error throuhg the obj, so we release the ref.
                  m_error->decref();
                  m_error = 0;
                  m_regB.setObject( obj );
                  popTry( true );
                  m_event = eventNone;
                  continue;
               }
               else {
                  // panic. Should not happen
                  if( m_errhand != 0 ) {
                     Error *err = new CodeError( ErrorParam( e_undef_sym, __LINE__ ).
                        module( "core.vm" ).symbol( "vm_run" ).extra( m_error->className() ) );
                     m_error->decref();
                     m_error = err;
                     m_errhand->handleError( err );
                  }
                  return;
               }
            }
            // we couldn't catch the error (this also means we're at m_stackBase zero)
            // we should handle it then exit
            else {
               // we should manage the error; if we're here, m_stackBase is zero,
               // so we are the last in charge
               if( m_errhand != 0 )
                  m_errhand->handleError( m_error );
               // we're out of business.
               return;
            }
         break;

         case eventYield:
            m_pc = m_pc_next;
            yield( m_yieldTime );
            if ( m_event == eventSleep )
               return;
            else if ( m_event == eventRisen )
            {
               m_pc = i_pc_redo_request;
               continue;
            }
            m_event = eventNone;
         continue;

         // this can only be generated by an electContext or rotateContext that finds there's the need to sleep
         // as contexts cannot be elected in atomic mode, and as wait event is
         // already guarded against atomic mode breaks, we let this through
         case eventSleep:
            return;

         case eventWait:
            if ( m_atomicMode )
            {
               raiseError( new InterruptedError( ErrorParam( e_interrupted ).origin( e_orig_vm ).
                     symbol( "vm_run" ).
                     module( "core.vm" ).
                     line( __LINE__ ).
                     hard() ) );
               // just re-parse the event
               m_pc = i_pc_redo_request;
               continue;
            }
            if ( m_sleepingContexts.empty() && !m_sleepAsRequests && m_yieldTime < 0.0 )
            {
               m_error = new GenericError( ErrorParam( e_deadlock ).origin( e_orig_vm ) );
               fillErrorContext( m_error );
               m_event = eventRisen;
               if( m_errhand != 0 )
                     m_errhand->handleError( m_error );
               return;
            }

            m_pc = m_pc_next;

            m_currentContext->save( this );

			// if wait time is > 0, put at sleep
			if( m_yieldTime > 0.0 )
				putAtSleep( m_currentContext, m_yieldTime );

			electContext();

            if ( m_event == eventSleep )
               return;

            m_event = eventNone;
         continue;

         case eventQuit:
            m_pc = m_pc_next;
            // quit the machine
         return;

         default:
            // switch to next instruction
            m_pc = m_pc_next;
      }

   } // end while -- VM LOOP
}

/****************************************************
   Op code handlers
*****************************************************/

// 0
void opcodeHandler_END( register VMachine *vm )
{
   // scan the contexts and remove the current one.
   if ( vm->m_sleepingContexts.empty() )
   {
      vm->m_event = VMachine::eventQuit;
      // nil also the A register
      vm->regA().setNil();
   }
   else {
      ListElement *iter = vm->m_contexts.begin();
      while( iter != 0 ) {
         if( iter->data() == vm->m_currentContext ) {
            vm->m_contexts.erase( iter );
			// removing the context also deletes it.

			// Not necessary, but we do for debug reasons (i.e. if we access it before election, we crash)
			vm->m_currentContext = 0;

			break;
         }
         iter = iter->next();
      }

      vm->electContext();
   }
}


// 1
void opcodeHandler_NOP( register VMachine *vm )
{
}

// 2
void opcodeHandler_PSHN( register VMachine *vm )
{
   vm->m_stack->resize( vm->m_stack->size() + 1 );
}

// 3
void opcodeHandler_RET( register VMachine *vm )
{
   vm->callReturn();
   vm->retnil();
}

// 4
void opcodeHandler_RETA( register VMachine *vm )
{
   vm->callReturn();

   //? Check this -- I don't think is anymore necessary
   if( vm->m_regA.type() == FLC_ITEM_REFERENCE )
      vm->m_regA.copy( vm->m_regA.asReference()->origin() );
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
   vm->m_regA = *vm->getOpcodeParam( 1 )->dereference();
   vm->callReturn();
}

// 8
void opcodeHandler_BOOL( register VMachine *vm )
{
   vm->m_regA.setBoolean( vm->getOpcodeParam( 1 )->dereference()->isTrue() );
}

// 9
void opcodeHandler_JMP( register VMachine *vm )
{
   vm->m_pc_next = vm->getNextNTD32();
}

// 0A
void opcodeHandler_GENA( register VMachine *vm )
{
   register uint32 size = (uint32) vm->getNextNTD32();
   CoreArray *array = new CoreArray( vm, size );

   // copy the m-topmost items in the stack into the array
   Item *data = array->elements();
   int32 base = vm->m_stack->size() - size;

   for ( uint32 i = 0; i < size; i++ ) {
      data[ i ] = vm->m_stack->itemAt(i + base);
   }
   array->length( size );
   vm->m_stack->resize( base );
   vm->m_regA.setArray( array );
}

// 0B
void opcodeHandler_GEND( register VMachine *vm )
{
   register uint32 length = (uint32) vm->getNextNTD32();
   LinearDict *dict = new LinearDict( vm, length );

   // copy the m-topmost items in the stack into the array
   int32 base = vm->m_stack->size() - ( length * 2 );
   for ( register uint32 i = base; i < vm->m_stack->size(); i += 2 ) {
      dict->insert( vm->stackItem(i), vm->stackItem(i+1));
   }
   vm->m_stack->resize( base );
   vm->m_regA.setDict( dict );
}


// 0C
void opcodeHandler_PUSH( register VMachine *vm )
{
   /** \TODO Raise a stack overflow error on VM stack boundary limit. */
   Item *data = vm->getOpcodeParam( 1 )->dereference();
   vm->m_stack->push( data );
}

// 0D
void opcodeHandler_PSHR( register VMachine *vm )
{
   Item *referenced = vm->getOpcodeParam( 1 );
   if ( ! referenced->isReference() )
   {
      GarbageItem *ref = new GarbageItem( vm, *referenced );
      referenced->setReference( ref );
   }

   vm->m_stack->push( referenced );
}


// 0E
void opcodeHandler_POP( register VMachine *vm )
{
   if ( vm->m_stack->size() == vm->m_stackBase ) {
      vm->raiseError( e_stackuf, "POP" );
      return;
   }
   //  --- WARNING: do not dereference!
   vm->getOpcodeParam( 1 )->copy( vm->m_stack->topItem() );
   vm->m_stack->pop();
}


// 0F
void opcodeHandler_INC( register VMachine *vm )
{
   Item *operand =  vm->getOpcodeParam( 1 )->dereference();

   switch( operand->type() )
   {
      case FLC_ITEM_INT: operand->setInteger( operand->asInteger() + 1 ); break;
      case FLC_ITEM_NUM: operand->setNumeric( operand->asNumeric() + 1.0 ); break;
      default:
         vm->raiseError( e_invop, "INC" );
   }
}

// 10
void opcodeHandler_DEC( register VMachine *vm )
{
   Item *operand =  vm->getOpcodeParam( 1 )->dereference();

   switch( operand->type() )
   {
      case FLC_ITEM_INT: operand->setInteger( operand->asInteger() - 1 ); break;
      case FLC_ITEM_NUM: operand->setNumeric( operand->asNumeric() - 1.0 ); break;
      default:
         vm->raiseError( e_invop, "DEC" );
   }
}


// 11
void opcodeHandler_NEG( register VMachine *vm )
{
   Item *operand = vm->getOpcodeParam( 1 )->dereference();

   switch( operand->type() )
   {
      case FLC_ITEM_INT: vm->m_regA.setInteger( -operand->asInteger() ); break;
      case FLC_ITEM_NUM: vm->m_regA.setNumeric( -operand->asNumeric() ); break;
      default:
         vm->raiseError( e_invop, "NEG" );
   }
}


// 12
void opcodeHandler_NOT( register VMachine *vm )
{
   vm->m_regA.setInteger( vm->getOpcodeParam( 1 )->dereference()->isTrue() ? 0 : 1 );
}


//13
void opcodeHandler_TRAL( register VMachine *vm )
{
   uint32 pcNext = vm->getNextNTD32();

   if ( vm->m_stack->size() < 3 ) {
      vm->raiseError( e_stackuf, "TRAL" );
      return;
   }

   register uint32 size = vm->m_stack->size();
   Item *iterator = vm->m_stack->itemPtrAt( size - 3 );
   Item *source = vm->m_stack->itemPtrAt( size - 1 );

   switch( source->type() )
   {
      case FLC_ITEM_ARRAY:
         if ( iterator->asInteger() + 1 >= source->asArray()->length()  )
            vm->m_pc_next = pcNext;
      break;

      case FLC_ITEM_DICT:
      case FLC_ITEM_OBJECT:
      case FLC_ITEM_ATTRIBUTE:
      {
         CoreIterator *iter = (CoreIterator *) iterator->asObject()->getUserData();
         if ( ! iter->hasNext() )
            vm->m_pc_next = pcNext;
      }
      break;

      case FLC_ITEM_STRING:
         if ( iterator->asInteger() + 1 >= source->asString()->length() )
            vm->m_pc_next = pcNext;
      break;

      case FLC_ITEM_MEMBUF:
         if ( iterator->asInteger() + 1 >= source->asMemBuf()->length() )
            vm->m_pc_next = pcNext;
      break;

      case FLC_ITEM_RANGE:
         if ( source->asRangeIsOpen() )
         {
            vm->m_pc_next = pcNext;
         }
         if ( source->asRangeStart() < source->asRangeEnd() )
         {
            if ( iterator->asInteger() + 1 >= source->asRangeEnd() )
               vm->m_pc_next = pcNext;
         }
         else {
            if ( iterator->asInteger() <= source->asRangeEnd() )
               vm->m_pc_next = pcNext;
         }
      break;

      default:
         vm->raiseError( e_invop, "TRAL" );
         return;
   }
}



//14
void opcodeHandler_IPOP( register VMachine *vm )
{
   register uint32 amount = (uint32) vm->getNextNTD32();
   if ( vm->m_stack->size() < amount ) {
      vm->raiseError( e_stackuf, "IPOP" );
      return;
   }

   vm->m_stack->resize( vm->m_stack->size() - amount );
}

//15
void opcodeHandler_XPOP( register VMachine *vm )
{
   Item *operand = vm->getOpcodeParam( 1 )->dereference();
   // use copy constructor.
   Item itm( *operand );
   operand->copy( vm->m_stack->topItem() );
   vm->m_stack->topItem().copy( itm );
}

//16
void opcodeHandler_GEOR( register VMachine *vm )
{
   vm->regA().setRange( (int32) vm->getOpcodeParam( 1 )->dereference()->forceInteger(), 0 , true );
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
   vm->m_pc_next = target;
}

//19
void opcodeHandler_RIS( register VMachine *vm )
{
   vm->m_regB = *vm->getOpcodeParam( 1 )->dereference();
   vm->m_event = VMachine::eventRisen;
}

//1A
void opcodeHandler_BNOT( register VMachine *vm )
{
   register Item *operand = vm->getOpcodeParam( 1 )->dereference();
   if ( operand->type() == FLC_ITEM_INT ) {
      vm->m_regA.setInteger( ~operand->asInteger() );
   }
   else
      vm->raiseError( e_bitwise_op, "BNOT" );
}

//1B
void opcodeHandler_NOTS( register VMachine *vm )
{
   register Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( operand1->isOrdinal() )
      operand1->setInteger( ~operand1->forceInteger() );
   else
      vm->raiseError( e_bitwise_op, "NOTS" );
}

//1c
void opcodeHandler_PEEK( register VMachine *vm )
{
   register Item *operand = vm->getOpcodeParam( 1 )->dereference();

   if ( vm->m_stack->size() == 0 ) {
      vm->raiseError( e_stackuf, "PEEK" );
      return;
   }
   *operand = vm->m_stack->topItem();
}

// 1D
void opcodeHandler_FORK( register VMachine *vm )
{
   uint32 pSize = (uint32) vm->getNextNTD32();
   uint32 pJump = (uint32) vm->getNextNTD32();

   VMContext *ctx = new VMContext( vm );

   if ( pSize > 0 ) {
      ctx->getStack()->reserve( pSize );
      for( uint32 i = 0; i < pSize; i++ ) {
         Item temp = vm->m_stack->itemAt( vm->m_stack->size() - pSize + i );
         ctx->getStack()->push( &temp );
      }
      vm->m_stack->resize( vm->m_stack->size() - pSize );
   }
   vm->m_contexts.pushBack( ctx );
   vm->putAtSleep( ctx, 0.0 );

   vm->m_pc = pJump;
   vm->m_pc_next = pJump;
}

// 1D - Missing

// 1E
void opcodeHandler_LD( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand2->isString() )
      operand1->setString( new GarbageString( vm, *operand2->asString() ) );
   else
      operand1->copy( *operand2 );
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
         gitem = new GarbageItem( vm, *operand2 );
         operand2->setReference( gitem );
      }

      operand1->setReference( gitem );
   }
}

// 20
void opcodeHandler_ADD( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         vm->m_regA.setInteger( operand1->asInteger() + operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         vm->m_regA.setNumeric( operand1->asInteger() + operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         vm->m_regA.setNumeric( operand1->asNumeric() + operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         vm->m_regA.setNumeric( operand1->asNumeric() + operand2->asNumeric() );
      return;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_INT:
      case FLC_ITEM_STRING<< 8 | FLC_ITEM_NUM:
      {
         int64 chr = operand2->forceInteger();
         if ( chr >= 0 && chr <= (int64) 0xFFFFFFFF )
         {
            GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
            gcs->append( (uint32) chr );
            vm->m_regA.setString( gcs );
            return;
         }
      }
      break;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_STRING:
      {
         GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
         gcs->append( *operand2->asString() );
         vm->m_regA.setString( gcs );
      }
      return;

      case FLC_ITEM_DICT<< 8 | FLC_ITEM_DICT:
      {
         CoreDict *dict = new LinearDict( vm, operand1->asDict()->length() + operand2->asDict()->length() );
         dict->merge( *operand1->asDict() );
         dict->merge( *operand2->asDict() );
         vm->m_regA.setDict( dict );
      }
      return;
   }

   // add any item to the end of an array.
   if( operand1->type() == FLC_ITEM_ARRAY )
   {
      CoreArray *first = operand1->asArray()->clone();

      if ( operand2->type() == FLC_ITEM_ARRAY ) {
         first->merge( *operand2->asArray() );
      }
      else {
         if ( operand2->isString() && operand2->asString()->garbageable() )
            first->append( operand2->asString()->clone() );
         else
            first->append( *operand2 );
      }
      vm->retval( first );
      return;
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("ADD").origin( e_orig_vm ) ) );
}

// 21
void opcodeHandler_SUB( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         vm->m_regA.setInteger( operand1->asInteger() - operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         vm->m_regA.setNumeric( operand1->asInteger() - operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         vm->m_regA.setNumeric( operand1->asNumeric() - operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         vm->m_regA.setNumeric( operand1->asNumeric() - operand2->asNumeric() );
      return;
   }

   // remove any element from an array
   if ( operand1->isArray() )
   {
      CoreArray *source = operand1->asArray();
      CoreArray *dest = source->clone();

      // if we have an array, remove all of it
      if( operand2->isArray() )
      {
         CoreArray *removed = operand2->asArray();
         for( uint32 i = 0; i < removed->length(); i ++ )
         {
            int32 rem = dest->find( removed->at(i) );
            if( rem >= 0 )
               dest->remove( rem );
         }
      }
      else {
         int32 rem = dest->find( *operand2 );
         if( rem >= 0 )
            dest->remove( rem );
      }

      // never raise.
      vm->m_regA = dest;
      return;
   }
   // remove various keys from arrays
   else if( operand1->isDict() )
   {
      CoreDict *source = operand1->asDict();
      CoreDict *dest = source->clone();

      // if we have an array, remove all of it
      if( operand2->isArray() )
      {
         CoreArray *removed = operand2->asArray();
         for( uint32 i = 0; i < removed->length(); i ++ )
         {
            dest->remove( removed->at(i) );
         }
      }
      else {
         dest->remove( *operand2 );
      }

      // never raise.
      vm->m_regA = dest;
      return;
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("SUB").origin( e_orig_vm ) ) );
}

// 22
void opcodeHandler_MUL( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         vm->m_regA.setInteger( operand1->asInteger() * operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         vm->m_regA.setNumeric( operand1->asInteger() * operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         vm->m_regA.setNumeric( operand1->asNumeric() * operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         vm->m_regA.setNumeric( operand1->asNumeric() * operand2->asNumeric() );
      return;
   }
   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MUL").origin( e_orig_vm ) ) );
}

// 23
void opcodeHandler_DIV( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand2->type() )
   {
      case FLC_ITEM_INT:
      {
         int64 val2 = operand2->asInteger();
         if ( val2 == 0 ) {
            vm->raiseRTError( new MathError( ErrorParam( e_div_by_zero ).origin( e_orig_vm ) ) );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               vm->m_regA.setNumeric( operand1->asInteger() / (numeric)val2 );
            return;
            case FLC_ITEM_NUM:
               vm->m_regA.setNumeric( operand1->asNumeric() / (numeric)val2 );
            return;
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         numeric val2 = operand2->asNumeric();
         if ( val2 == 0.0 ) {
            vm->raiseRTError( new MathError( ErrorParam( e_div_by_zero ).origin( e_orig_vm ) ) );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               vm->m_regA.setNumeric( operand1->asInteger() / (numeric)val2 );
            return;
            case FLC_ITEM_NUM:
               vm->m_regA.setNumeric( operand1->asNumeric() / (numeric)val2 );
            return;
         }
      }
      break;
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("DIV").origin( e_orig_vm ) ) );
}


//24
void opcodeHandler_MOD( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->type() == FLC_ITEM_INT && operand2->type() == FLC_ITEM_INT ) {
      if ( operand2->asInteger() == 0 )
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MOD").origin( e_orig_vm ) ) );
      else
         vm->m_regA.setInteger( operand1->asInteger() % operand2->asInteger() );
   }
   else if ( operand1->isOrdinal() && operand2->isOrdinal() )
   {
      if ( operand2->forceNumeric() == 0.0 )
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MOD").origin( e_orig_vm ) ) );
      else
         vm->regA().setNumeric( fmod( operand1->forceNumeric(), operand2->forceNumeric() ) );
   }
   else
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MOD").origin( e_orig_vm ) ) );
}

// 25
void opcodeHandler_POW( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   numeric powval;

   errno = 0;
   switch( operand2->type() )
   {
      case FLC_ITEM_INT:
      {
         if ( operand2->asInteger() == 0 ) {
            vm->retval( (int64) 1 );
            return;
         }

         numeric val2 = (numeric) operand2->asInteger();
         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               powval = pow( (double)operand1->asInteger(), val2 );
            break;

            case FLC_ITEM_NUM:
               powval = pow( operand1->asNumeric(), val2 );
            break;

            default:
               vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("POW").origin( e_orig_vm ) ) );
               return;
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         numeric val2 = operand2->asNumeric();
         if ( val2 == 0.0 ) {
            vm->retval( (int64) 1 );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               powval = pow( (double) operand1->asInteger(), val2 );
            break;

            case FLC_ITEM_NUM:
               powval = pow( operand1->asNumeric(), val2 );
            break;

            default:
               vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("POW").origin( e_orig_vm ) ) );
               return;
         }
      }
      break;

      default:
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("POW").origin( e_orig_vm ) ) );
         return;
   }

   if ( errno != 0  )
   {
      vm->raiseRTError( new MathError( ErrorParam( e_domain ).origin( e_orig_vm ) ) );
   }
   else {
      vm->retval( powval );
   }
}

// 26
void opcodeHandler_ADDS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         operand1->setInteger(operand1->asInteger() + operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         operand1->setNumeric(operand1->asInteger() + operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         operand1->setNumeric(operand1->asNumeric() + operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         operand1->setNumeric(operand1->asNumeric() + operand2->asNumeric() );
      return;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_INT:
      case FLC_ITEM_STRING<< 8 | FLC_ITEM_NUM:
      {
         int64 chr = operand2->forceInteger();
         if ( chr >= 0 && chr <= (int64) 0xFFFFFFFF )
         {
            String *str = new GarbageString( vm, *operand1->asString() );
            str->append( (uint32) chr );
            operand1->setString( str );
            return;
         }
      }
      break;

      case FLC_ITEM_STRING<< 8 | FLC_ITEM_STRING:
      {
         String *str = new GarbageString( vm, *operand1->asString() );
         str->append( *operand2->asString() );
         operand1->setString( str );
      }
      return;

      case FLC_ITEM_DICT<< 8 | FLC_ITEM_DICT:
      {
         operand1->asDict()->merge( *operand2->asDict() );
      }
      return;

   }

   // add any item to the end of an array.
   if( operand1->type() == FLC_ITEM_ARRAY )
   {
      if ( operand2->type() == FLC_ITEM_ARRAY ) {
         operand1->asArray()->merge( *operand2->asArray() );
      }
      else {
         if ( operand2->isString() && operand2->asString()->garbageable() )
            operand1->asArray()->append( operand2->asString()->clone() );
         else
            operand1->asArray()->append( *operand2 );
      }
      return;
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("ADDS").origin( e_orig_vm ) ) );
}

//27
void opcodeHandler_SUBS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         operand1->setInteger(operand1->asInteger() - operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         operand1->setNumeric(operand1->asInteger() - operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         operand1->setNumeric(operand1->asNumeric() - operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         operand1->setNumeric(operand1->asNumeric() - operand2->asNumeric() );
      return;
   }


   // remove any element from an array
   if ( operand1->isArray() )
   {
      CoreArray *source = operand1->asArray();

      // if we have an array, remove all of it
      if( operand2->isArray() )
      {
         CoreArray *removed = operand2->asArray();
         for( uint32 i = 0; i < removed->length(); i ++ )
         {
            int32 rem = source->find( removed->at(i) );
            if( rem >= 0 )
               source->remove( rem );
         }
      }
      else {
         int32 rem = source->find( *operand2 );
         if( rem >= 0 )
            source->remove( rem );
      }

      // never raise.
      return;
   }
   // remove various keys from arrays
   else if( operand1->isDict() )
   {
      CoreDict *source = operand1->asDict();

      // if we have an array, remove all of it
      if( operand2->isArray() )
      {
         CoreArray *removed = operand2->asArray();
         for( uint32 i = 0; i < removed->length(); i ++ )
         {
            source->remove( removed->at(i) );
         }
      }
      else {
         source->remove( *operand2 );
      }

      // never raise.
      vm->m_regA = source;
      return;
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("SUBS").origin( e_orig_vm ) ) );
}


//28

void opcodeHandler_MULS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_INT << 8 | FLC_ITEM_INT:
         operand1->setInteger(operand1->asInteger() * operand2->asInteger() );
      return;

      case FLC_ITEM_INT << 8 | FLC_ITEM_NUM:
         operand1->setNumeric(operand1->asInteger() * operand2->asNumeric() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_INT:
         operand1->setNumeric(operand1->asNumeric() * operand2->asInteger() );
      return;

      case FLC_ITEM_NUM<< 8 | FLC_ITEM_NUM:
         operand1->setNumeric(operand1->asNumeric() * operand2->asNumeric() );
      return;
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MULS").origin( e_orig_vm ) ) );
}

//29
void opcodeHandler_DIVS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand2->type() )
   {
      case FLC_ITEM_INT:
      {
         int64 val2 = operand2->asInteger();
         if ( val2 == 0 ) {
            vm->raiseRTError( new MathError( ErrorParam( e_div_by_zero ).origin( e_orig_vm ) ) );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               operand1->setNumeric( operand1->asInteger() / (numeric)val2 );
            return;
            case FLC_ITEM_NUM:
               operand1->setNumeric( operand1->asNumeric() / (numeric)val2 );
            return;
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         numeric val2 = operand2->asNumeric();
         if ( val2 == 0.0 ) {
            vm->raiseRTError( new MathError( ErrorParam( e_div_by_zero ).origin( e_orig_vm ) ) );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               operand1->setNumeric( operand1->asInteger() / (numeric)val2 );
            return;
            case FLC_ITEM_NUM:
               operand1->setNumeric( operand1->asNumeric() / (numeric)val2 );
            return;
         }
      }
      break;
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("DIVS").origin( e_orig_vm ) ) );
}


//2A
void opcodeHandler_MODS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->type() == FLC_ITEM_INT && operand2->type() == FLC_ITEM_INT ) {
      if ( operand2->asInteger() == 0 )
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MODS").origin( e_orig_vm ) ) );
      else
         operand1->setInteger( operand1->asInteger() % operand2->asInteger() );
   }
   else if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      if ( operand2->forceNumeric() == 0.0 )
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MODS").origin( e_orig_vm ) ) );
      else
         operand1->setNumeric( fmod( operand1->forceNumeric(), operand2->forceNumeric() ) );
   }
   else
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("MODS").origin( e_orig_vm ) ) );
}

//2B
void opcodeHandler_BAND( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      vm->m_regA.setInteger( operand1->forceInteger() & operand2->forceInteger() );
   }
   else
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("BAND").origin( e_orig_vm ) ) );
}

//2C
void opcodeHandler_BOR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      vm->m_regA.setInteger( operand1->forceInteger() | operand2->forceInteger() );
   }
   else
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("BOR").origin( e_orig_vm ) ) );
}

//2D
void opcodeHandler_BXOR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isOrdinal() && operand2->isOrdinal() ) {
      vm->m_regA.setInteger( operand1->forceInteger() ^ operand2->forceInteger() );
   }
   else
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("BXOR").origin( e_orig_vm ) ) );
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
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("ANDS").origin( e_orig_vm ) ) );
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
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("ORS").origin( e_orig_vm ) ) );
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
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("XORS").origin( e_orig_vm ) ) );
}

//31
void opcodeHandler_GENR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->m_regA.setRange(
         (int32) operand1->forceInteger(),
         (int32) operand2->forceInteger(), false );
}

//32
void opcodeHandler_EQ( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->m_regA.setBoolean( vm->compareItems( *operand1, *operand2 ) == 0 );
}

//33
void opcodeHandler_NEQ( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->m_regA.setBoolean( vm->compareItems( *operand1, *operand2 ) != 0 );
}

//34
void opcodeHandler_GT( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->regA().setBoolean( vm->compareItems( *operand1, *operand2 ) > 0 );
}

//35
void opcodeHandler_GE( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->regA().setBoolean( vm->compareItems( *operand1, *operand2 ) >= 0 );
}

//36
void opcodeHandler_LT( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->regA().setBoolean( vm->compareItems( *operand1, *operand2 ) < 0 );
}

//37
void opcodeHandler_LE( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->regA().setBoolean( vm->compareItems( *operand1, *operand2 ) <= 0 );
}

//38
void opcodeHandler_IFT( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand2->isTrue() )
      vm->m_pc_next = pNext;
}

//39
void opcodeHandler_IFF( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( ! operand2->isTrue() )
      vm->m_pc_next = pNext;
}

//3A
void opcodeHandler_CALL( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( ! vm->callItem( *operand2, pNext, VMachine::e_callFrame ) )
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("CALL").origin( e_orig_vm ) ) );
}

//3B
void opcodeHandler_INST( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand2->type() != FLC_ITEM_CLASS )
   {
      vm->raiseError( e_invop, "INST" );
      return;
   }

   Symbol *cls = operand2->asClass()->symbol();
   Symbol *ctor = cls->getClassDef()->constructor();
   if ( ctor != 0 ) {
      if( ! vm->callItem( *operand2, pNext, VMachine::e_callInstFrame ) )
         vm->raiseError( e_invop, "INST" );
   }
}

//3C
void opcodeHandler_ONCE( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   Symbol *call = 0;

   if ( operand2->isFunction() )
   {
      call = operand2->asFunction();
   }
   else if ( operand2->isMethod() )
   {
      call = operand2->asMethodFunction();
   }

   if ( call != 0 && call->isFunction() )
   {
      // we suppose we're in the same module as the function things we are...
      register uint32 itemId = call->getFuncDef()->onceItemId();
      if ( vm->moduleItem( itemId ).isNil() )
         vm->moduleItem( itemId ).setInteger( 1 );
      else
         vm->m_pc_next = pNext;
      return;
   }

   vm->raiseError( e_invop, "ONCE" );
}

//3D
void opcodeHandler_LDV( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_STRING << 8 | FLC_ITEM_INT:
      {
         int32 pos = (int32) operand2->asInteger();
         String *cs = operand1->asString();
         if ( cs->checkPosBound( pos ) )
         {
            vm->retval( new GarbageString( vm, String(*cs, pos, pos+1 ) ) );
            return;
         }
      }
      break;

      case FLC_ITEM_STRING << 8 | FLC_ITEM_NUM:
      {
         int32 pos = (int32) operand2->asNumeric();
         String *cs = operand1->asString();
         if ( cs->checkPosBound( pos ) )  {
            vm->retval( new GarbageString( vm, String(*cs, pos, pos+1 ) ) );
            return;
         }
      }
      break;

      case FLC_ITEM_MEMBUF << 8 | FLC_ITEM_INT:
      case FLC_ITEM_MEMBUF << 8 | FLC_ITEM_NUM:
      {
         int64 pos = (int64) operand2->forceInteger();
         MemBuf *mb = operand1->asMemBuf();
         if ( pos >= 0 && pos < (int64) mb->length() )  {
            vm->retval( (int64) mb->get( pos ) );
            return;
         }
      }
      break;

      case FLC_ITEM_STRING << 8 | FLC_ITEM_RANGE:
      {
         String *cs = operand1->asString();
         int32 rstart =  operand2->asRangeStart();
         if ( operand2->asRangeIsOpen() )
         {
            if ( cs->checkPosBound( rstart ) ) {
               vm->retval( new GarbageString( vm, String( *cs, rstart ) ) );
            }
            else {
               vm->retval( new GarbageString( vm ) );
            }
            return;
         }
         else {
            int32 rend =  operand2->asRangeEnd();
            if ( cs->checkRangeBound( rstart, rend ) )
            {
               vm->retval( new GarbageString( vm, String(*cs, rstart, rend ) ) );
               return;
            }
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_INT:
      {
         register int32 pos = (int32) operand2->asInteger();
         CoreArray *array = operand1->asArray();
         if (!( -pos > int(array->length()) || pos >= int(array->length()) ) )
         {
            vm->retval( (*array)[ pos ] );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_NUM:
      {
         register int32 pos = (int32) operand2->asNumeric();
         CoreArray *array = operand1->asArray();
         if ( ! (-pos > int(array->length()) || pos >= int(array->length()) ) ){
            vm->retval( (*array)[ pos ] );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_RANGE:
      {
         CoreArray *array =  operand1->asArray();

         // open ranges?
         if ( operand2->asRangeIsOpen() &&
              (operand2->asRangeStart() >= 0 && (int) array->length() <= operand2->asRangeStart() ) ||
              (operand2->asRangeStart() < 0 && (int) array->length() < -operand2->asRangeStart() )
              )
         {
            vm->retval( new CoreArray( vm ) );
            return;
         }

         register int32 end = operand2->asRangeIsOpen() ? array->length() : operand2->asRangeEnd();
         array = array->partition( operand2->asRangeStart(), end );
         if ( array != 0 )
         {
            vm->retval( array );
            return;
         }
      }
      break;

      case FLC_ITEM_RANGE << 8 | FLC_ITEM_INT:
      case FLC_ITEM_RANGE << 8 | FLC_ITEM_NUM:
      {
         int32 pos = (int32) operand2->forceInteger();
         switch( pos )
         {
            case 0: vm->retval( operand1->asRangeStart() ); return;

            case 1: case -1:
               if( operand1->asRangeIsOpen() )
                  vm->retnil();
               else
                  vm->retval( operand1->asRangeEnd() );
            return;
         }
      }
      break;

      default:
         if( operand1->type() == FLC_ITEM_DICT ) {
            if( operand1->asDict()->find( *operand2, vm->m_regA ) ) {
               return;
            }
         }
   }

   vm->raiseRTError(
            new RangeError( ErrorParam( e_arracc ).origin( e_orig_vm ).extra( "LDV" ) ) );
}

//3E
void opcodeHandler_LDP( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand2->isString() )
   {
      String *property = operand2->asString();

      Item *source = operand1;
      CoreObject *self = vm->m_regS1.isNil()? 0: vm->m_regS1.asObject();
      CoreClass *sourceClass=0;
      Item prop;
      uint32 pos;

      switch( source->type() )
      {
         case FLC_ITEM_OBJECT:
            if( source->asObject()->getProperty( *property, prop ) ) {
               // we must create a method if the property is a function.
               Item *p = prop.dereference();
               switch( p->type() ) {
                  case FLC_ITEM_FUNC:
                     // the function may be a dead function; by so, the method will become a dead method,
                     // and it's ok for us.
                     vm->m_regA.setMethod( source->asObject(), p->asFunction(), p->asModule() );
                  break;

                  case FLC_ITEM_CLASS:
                     vm->m_regA.setClassMethod( source->asObject(), p->asClass() );
                  break;
                  default:
                     vm->m_regA = *p;
               }
               return;
            }
         break;

         case FLC_ITEM_CLSMETHOD:
            sourceClass = source->asMethodClass();
            self = source->asMethodObject();

         // do not break: fallback
         case FLC_ITEM_CLASS:
            if ( sourceClass == 0 )
               sourceClass = source->asClass();

            if( sourceClass->properties().findKey( property, pos ) )
            {
               Item *prop = sourceClass->properties().getValue( pos );
               // now, accessing a method in a class means that we want to call the base method in a
               // self item:
               if( prop->type() == FLC_ITEM_FUNC )
               {
                  if ( self != 0 )
                     vm->m_regA.setMethod( self, prop->asFunction(), prop->asModule() );
                  else
                     vm->m_regA.setFunction( prop->asFunction(), prop->asModule() );
               }
               else
               {
                  vm->regA() = *prop;
               }
               return;
            }
         break;
      }

      // try to find a generic method
      if( source->getBom( *property, vm->regA(), vm->m_fbom ) )
         return;
   }

   vm->raiseRTError(
      new RangeError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
         extra( operand2->isString() ? *operand2->asString() : "?" ) ) );
}


// 3F
void opcodeHandler_TRAN( register VMachine *vm )
{
   uint32 p1 = vm->getNextNTD32();
   uint32 p2 = vm->getNextNTD32();
   uint32 p3 = vm->getNextNTD32();

   if ( vm->m_stack->size() < 3 ) {
      vm->raiseError( e_stackuf, "TRAN" );
      return;
   }

   register int size = vm->m_stack->size();
   Item *iterator = &vm->stackItem( size - 3 );
   bool isIterator = ! iterator->isInteger();
   Item *dest = &vm->stackItem( size - 2 );
   bool isDestStack = ! dest->isReference();
   Item *source = &vm->stackItem( size - 1 );

   switch( source->type() )
   {
      case FLC_ITEM_ARRAY:
      {
         if ( isIterator )
         {
            vm->raiseError( e_stackuf, "TRAN" );
            return;
         }

         CoreArray *sarr = source->asArray();
         uint32 counter = (uint32) iterator->asInteger();

         if( p3 == 1 )
         {
            if ( counter >= sarr->length() )
            {
               vm->raiseError( e_arracc, "TRAN" );
               return;
            }
            sarr->remove( counter );
         }
         else {
            counter++;
            //update counter
            iterator->setInteger( counter );
         }

         if ( counter >= sarr->length() ) {
            //we are done -- go on trough last element block and cleanup
            vm->m_pc_next = p2;
            return;
         }

         source = & (*sarr)[counter];
         if( isDestStack )
         {
            uint32 vars = (uint32) dest->asInteger();
            if ( ! source->isArray() || vars != source->asArray()->length() ) {
               vm->raiseRTError(
                  new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
               return;
            }

            CoreArray *sarr = source->asArray();
            for ( uint32 i = 0; i < vars; i ++ ) {
               vm->stackItem( size - 3 - vars + i ).dereference()->copy( (* sarr )[i] );
            }
         }
         else
            dest->dereference()->copy( *source );

      }
      break;

      case FLC_ITEM_DICT:
         if ( ! isIterator || ! isDestStack )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
            return;
         }
         else {
            DictIterator *iter = (DictIterator *) iterator->asObject()->getUserData();

            if( ! iter->isValid() )
            {
               vm->raiseError( e_arracc, "TRAN" );
               return;
            }

            if( p3 == 1 )
            {
               CoreDict *sdict = source->asDict();
               sdict->remove( *iter );
               if( ! iter->hasNext() )
               {
                  vm->m_pc_next = p2;
                  return;
               }
            }
            else {
               if( ! iter->next() )
               {
                  // great, we're done -- let the code pass through.
                  vm->m_pc_next = p2;
                  return;
               }
            }

            vm->stackItem( size -3 - 2 ).dereference()->
                  copy( iter->getCurrentKey() );
            vm->stackItem( size -3 - 1 ).dereference()->
                  copy( *iter->getCurrent().dereference() );
         }
      break;

      case FLC_ITEM_ATTRIBUTE:
      case FLC_ITEM_OBJECT:
         if ( ! isIterator )
         {
            vm->raiseError( e_arracc, "TRAN" );
            return;
         }
         else {
            CoreIterator *iter = (CoreIterator *) iterator->asObject()->getUserData();

            if( ! iter->isValid() )
            {
               vm->raiseError( e_arracc, "TRAN" );
               return;
            }

            if( p3 == 1 )
            {
               if( ! iter->erase() )
               {
                  vm->raiseError( e_arracc, "TRAN" );
                  return;
               }
               // had the delete invalidated this?
               if( ! iter->isValid() )
               {
                  vm->m_pc_next = p2;
                  return;
               }
            }
            else {
               if( ! iter->next() )
               {
                  // great, we're done -- let the code pass through.
                  vm->m_pc_next = p2;
                  return;
               }
            }

            *dest->dereference() = iter->getCurrent();
         }
      break;

      case FLC_ITEM_STRING:
         if ( isIterator || isDestStack )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
            return;
         }
         else {
            uint32 counter = (uint32) iterator->asInteger();
            String *sstr = source->asString();

            if( p3 == 1 )
            {
               sstr->remove( counter, 1 );
            }
            else {
               counter++;
               //update counter
               iterator->setInteger( counter );
            }

            if( counter >= sstr->length() ) {
               vm->m_pc_next = p2;
               return;
            }

            *dest->dereference() = new GarbageString( vm,
                  sstr->subString( counter, counter + 1 ) );
         }
      break;

      case FLC_ITEM_MEMBUF:
         if ( isIterator || isDestStack )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
            return;
         }
         else {
            uint32 counter = (uint32) iterator->asInteger();
            MemBuf *smb = source->asMemBuf();

            counter++;
            //update counter
            iterator->setInteger( counter );
            
            if( counter >= smb->length() ) {
               vm->m_pc_next = p2;
               return;
            }

            *dest->dereference() = (int64) smb->get(counter);
         }
      break;

      case FLC_ITEM_RANGE:
         if ( isIterator || isDestStack )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
            return;
         }
         else {
            int32 counter = (int32) iterator->asInteger();

            if ( source->asRangeIsOpen() ) {
               vm->m_pc_next = p2;
               return;
            }

            // if( p3 == 1 ) -- we Ignore this case, and let continue dropping to act as continue.

            if( source->asRangeStart() < source->asRangeEnd() )
            {
               if ( ++counter == source->asRangeEnd() ) {
                  vm->m_pc_next = p2;
                  return;
               }
            }
            else {
               if ( counter-- == source->asRangeEnd() ) {
                  vm->m_pc_next = p2;
                  return;
               }
            }

            //update counter
            iterator->setInteger( counter );
            *dest->dereference() = (int64) counter;
         }
      break;

      default:
         vm->raiseError( e_invop, "TRAN" );
         return;
   }

   // loop to begin.
   vm->m_pc_next = p1;
}

//40
void opcodeHandler_UNPK( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->type() != operand2->type() || operand1->type() != FLC_ITEM_ARRAY ) {
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("UNPK").origin( e_orig_vm ) ) );
      return;
   }

   CoreArray *source = operand2->asArray();
   CoreArray *dest = operand1->asArray();
   Item *s_elems = source->elements();
   Item *d_elems = dest->elements();
   uint32 len = source->length();

   if( len != dest->length() )
   {
      vm->raiseRTError(
         new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ) ) );
      return;
   }

   for ( uint32 i = 0; i < len; i++ )  {
      *d_elems[i].dereference() = *s_elems[i].dereference();
   }
}

//41
void opcodeHandler_SWCH( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();
   uint64 sw_count = (uint64) vm->getNextNTD64();

   byte *tableBase =  vm->m_code + vm->m_pc_next;

   uint16 sw_int = (int16) (sw_count >> 48);
   uint16 sw_rng = (int16) (sw_count >> 32);
   uint16 sw_str = (int16) (sw_count >> 16);
   uint16 sw_obj = (int16) sw_count;

   //determine the value type to be checked
   switch( operand2->type() )
   {
      case FLC_ITEM_NIL:
         if ( *reinterpret_cast<uint32 *>( tableBase ) != 0xFFFFFFFF ) {
            vm->m_pc_next = endianInt32( *reinterpret_cast<uint32 *>( tableBase )  );
            return;
         }
      break;

      case FLC_ITEM_INT:
         if ( sw_int > 0 &&
               vm->seekInteger( operand2->asInteger(), tableBase + sizeof(uint32), sw_int, vm->m_pc_next ) )
            return;
         if ( sw_rng > 0 &&
               vm->seekInRange( operand2->asInteger(),
                     tableBase + sizeof(uint32) + sw_int * (sizeof(uint64) + sizeof(uint32) ),
                     sw_rng, vm->m_pc_next ) )
            return;
      break;

      case FLC_ITEM_NUM:
         if ( sw_int > 0 &&
               vm->seekInteger( operand2->forceInteger(), tableBase + sizeof(uint32), sw_int, vm->m_pc_next ) )
            return;
         if ( sw_rng > 0 &&
               vm->seekInRange( operand2->forceInteger(),
                     tableBase + sizeof(uint32) + sw_int * (sizeof(uint64) + sizeof(uint32) ),
                     sw_rng, vm->m_pc_next ) )
            return;
      break;

      case FLC_ITEM_STRING:
         if ( sw_str > 0 &&
               vm->seekString( operand2->asString(),
                  tableBase + sizeof(uint32) +
                     sw_int * (sizeof(uint64) + sizeof(uint32) )+
                     sw_rng * (sizeof(uint64) + sizeof(uint32) ),
                     sw_str,
                     vm->m_pc_next ) )
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
               vm->m_pc_next ) )
            return;

   // in case of failure...
   vm->m_pc_next = pNext;
}

//42
void opcodeHandler_HAS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isObject() )
   {
      if ( operand2->isAttribute() )
      {
         vm->regA() = (int64) ( operand1->asObject()->has( operand2->asAttribute() ) ? 1: 0 );
         return;
      }
      else if ( operand2->isString() )
      {
         vm->regA() = (int64) ( operand1->asObject()->has( *operand2->asString() ) ? 1: 0 );
         return;
      }
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("HAS") ) );
}

//43
void opcodeHandler_HASN( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isObject() )
   {
      if ( operand2->isAttribute() )
      {
         vm->regA() = (int64) ( operand1->asObject()->has( operand2->asAttribute() ) ? 0: 1 );
         return;
      }
      else if ( operand2->isString() )
      {
         vm->regA() = (int64) ( operand1->asObject()->has( *operand2->asString() ) ? 0: 1 );
         return;
      }
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("HASN") ) );
}

//44
void opcodeHandler_GIVE( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isObject() )
   {
      if ( operand2->isAttribute() )
      {
         operand2->asAttribute()->giveTo( operand1->asObject() );
         return;
      }
      else if ( operand2->isString() )
      {
         Attribute *attrib = vm->findAttribute( *operand2->asString() );
         if ( attrib != 0 )
         {
            attrib->giveTo( operand1->asObject() );
         }
         return;
      }
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("GIVE") ) );
}

//45
void opcodeHandler_GIVN( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isObject() )
   {
      if ( operand2->isAttribute() )
      {
         operand2->asAttribute()->removeFrom( operand1->asObject() );
         return;
      }
      else if ( operand2->isString() )
      {
         Attribute *attrib = vm->findAttribute( *operand2->asString() );
         if ( attrib != 0 )
         {
            attrib->removeFrom( operand1->asObject() );
         }
         return;
      }
   }

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("GIVN") ) );
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
         Item *elements =  operand2->asArray()->elements();
         for( uint32 pos = 0; pos <operand2->asArray()->length(); pos++ )
         {
            if( elements[pos].equal( *operand1 )) {
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
            result = operand2->asObject()->hasProperty( *operand1->asString() );
      break;

      case FLC_ITEM_CLASS:
      {
         if( operand1->type() == FLC_ITEM_STRING )
         {
            uint32 pos;
            result = operand2->asClass()->properties().findKey( operand1->asString(), pos ) != 0;
         }
      }
      break;

      case FLC_ITEM_METHOD:
         //TODO
      break;

   }

   vm->m_regA.setBoolean( result );
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
      vm->raiseError( e_invop, "PROV" ); // hard error, as the string should be created by the compiler
      return;
   }

   bool result;
   uint32 pos;
   switch ( operand1->type() ) {
      case FLC_ITEM_CLASS:
         result = operand1->asClass()->properties().findKey( operand2->asString(), pos ) != 0;
      break;

      case FLC_ITEM_OBJECT:
         result = operand1->asObject()->hasProperty( *operand2->asString() );
      break;


      default:
         result = false;
   }

   vm->regA().setBoolean( result );
}



//49
void opcodeHandler_STVS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if(  vm->m_stack->empty() )
   {
      vm->raiseError( e_stackuf, "STVS" );
      return;
   }

   Item origin = vm->m_stack->topItem();
   vm->m_stack->pop();


   // try to access a dictionary with every item
   // access addition.
   if( operand1->type() == FLC_ITEM_DICT ) {
      operand1->asDict()->insert( *(operand2), origin );
      return;
   }

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_STRING << 8 | FLC_ITEM_INT:
      case FLC_ITEM_STRING << 8 | FLC_ITEM_NUM:
         if ( origin.isString() ) {
            register int32 pos = (int32) operand2->forceInteger();
            String *cs_orig = origin.asString();
            if( cs_orig->length() > 0 ) {
               if ( pos < (int32) operand1->asString()->length() )
               {
                  GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
                  gcs->setCharAt( pos, cs_orig->getCharAt(0) );
                  operand1->setString( gcs );
                  return;
               }
            }
         }
      break;

      case FLC_ITEM_STRING << 8 | FLC_ITEM_RANGE:
         if( origin.isString() )
         {
            GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
            operand1->setString( gcs );

            bool result = operand2->asRangeIsOpen() ?
               gcs->change( operand2->asRangeStart(), *origin.asString() ) :
               gcs->change( operand2->asRangeStart(), operand2->asRangeEnd(), *origin.asString() );
            if ( result )
               return;
         }
      break;

      case FLC_ITEM_MEMBUF << 8 | FLC_ITEM_INT:
      case FLC_ITEM_MEMBUF << 8 | FLC_ITEM_NUM:
      {
         int64 pos = (int64) operand2->forceInteger();
         MemBuf *mb = operand1->asMemBuf();
         if ( pos >= 0 && pos < (int64) mb->length() && origin.isOrdinal() )
         {
            mb->set( pos, origin.forceInteger() );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_INT:
      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_NUM:
      {
         register int32 pos = (int32) operand2->forceInteger();
         CoreArray *array = operand1->asArray();
         if ( pos >= (-(int)array->length()) && pos < (int32) array->length() ) {
            (*array)[ pos ] = origin;
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_RANGE:
      {
         CoreArray *array =  operand1->asArray();
         register int32 end = operand2->asRangeIsOpen() ? array->length() : operand2->asRangeEnd();
         register int32 start = operand2->asRangeStart();
         if( origin.isArray() ) {
            if( array->change( *origin.asArray() , start, end ) )
               return;
         }
         else {
            if ( start != end ) {// insert
               if( ! array->remove( start, end ) )
                  break;
            }
            if( array->insert( origin, start ) )
               return;
         }
      }
      break;
   }

    vm->raiseRTError( new RangeError( ErrorParam( e_arracc ).origin( e_orig_vm ) ) );
}

//4A
void opcodeHandler_STPS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( vm->m_stack->empty() )
   {
      vm->raiseError( e_stackuf, "STPS" );
      return;
   }

   Item *target = operand1;
   Item *method = operand2;
   Item item = vm->m_stack->topItem();
   vm->m_stack->pop();

   if ( method->isString() && target->isObject() )
   {
      // Are we restoring an original item?
      if( item.isMethod() && item.asMethodObject() == target->asObject() )
      {
         item.setFunction( item.asMethodFunction(), item.asModule() );
      }
      else if ( item.isString() )
      {
         GarbageString *gcs = new GarbageString( vm, *item.asString() );
         item.setString( gcs );
      }

      if( target->asObject()->setProperty( *method->asString(), item ) )
         return;

      vm->raiseRTError(
         new RangeError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
            extra( *method->asString() ) ) );

      return;
   }

    vm->raiseError( e_prop_acc, "STPS" );
}

//4B
void opcodeHandler_AND( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->m_regA.setBoolean( operand1->isTrue() && operand2->isTrue() );
}

//4C
void opcodeHandler_OR( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   vm->m_regA.setBoolean( operand1->isTrue() || operand2->isTrue() );
}

//4E
void opcodeHandler_PASS( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();

   if( ! vm->callItemPass( *operand1 ) )
      vm->raiseError( e_invop, "PASS" );
}

//4F
void opcodeHandler_PSIN( register VMachine *vm )
{
   Item *operand1 =  vm->getOpcodeParam( 1 )->dereference();

   if( ! vm->callItemPassIn( *operand1 ) )
      vm->raiseError( e_invop, "PSIN" );
}

//50
void opcodeHandler_STV( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();
   Item *origin = vm->getOpcodeParam( 3 )->dereference();

   // try to access a dictionary with every item
   // access addition.
   if( operand1->type() == FLC_ITEM_DICT ) {
      if ( origin->isString() && origin->asString()->garbageable() )
         operand1->asDict()->insert( *(operand2), new GarbageString( vm, *origin->asString() ) );
      else
         operand1->asDict()->insert( *(operand2), *origin );
      return;
   }

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_STRING << 8 | FLC_ITEM_INT:
      case FLC_ITEM_STRING << 8 | FLC_ITEM_NUM:
         if ( origin->type() == FLC_ITEM_STRING ) {
            register int32 pos = (int32) operand2->forceInteger();
            String *cs_orig = origin->asString();
            if( cs_orig->length() > 0 ) {
               String *cs = operand1->asString();
               if ( cs->checkPosBound( pos ) ) {
                  cs->setCharAt( pos, cs_orig->getCharAt(0) );
                  return;
               }
            }
         }
         else if( origin->isOrdinal() )
         {
            int64 chr = origin->forceInteger();
            if ( chr >=0 && chr <= (int64) 0xFFFFFFFF )
            {
               String *cs = operand1->asString();
               register int32 pos = (int32) operand2->forceInteger();
               if ( cs->checkPosBound( pos ) ) {
                  cs->setCharAt( pos, (int32) chr );
                  return;
               }
            }
         }
      break;

      case FLC_ITEM_STRING << 8 | FLC_ITEM_RANGE:
         if( origin->type() == FLC_ITEM_STRING ) {
            GarbageString *gcs = new GarbageString( vm, *operand1->asString() );
            operand1->setString( gcs );
            bool result = operand2->asRangeIsOpen() ?
               gcs->change( operand2->asRangeStart(), *origin->asString() ) :
               gcs->change( operand2->asRangeStart(), operand2->asRangeEnd(), *origin->asString() );
            if ( result )
               return;
         }
      break;

      case FLC_ITEM_MEMBUF << 8 | FLC_ITEM_INT:
      case FLC_ITEM_MEMBUF << 8 | FLC_ITEM_NUM:
      {
         int64 pos = (int64) operand2->forceInteger();
         MemBuf *mb = operand1->asMemBuf();
         if ( pos >= 0 && pos < (int64) mb->length() && origin->isOrdinal() )
         {
            mb->set( pos, origin->forceInteger() );
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_INT:
      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_NUM:
      {
         register int32 pos = (int32) operand2->forceInteger();
         CoreArray *array = operand1->asArray();
         if ( pos >= (-(int)array->length()) && pos < (int32) array->length() ) {
            if ( origin->isString() && origin->asString()->garbageable() )
               (*array)[ pos ] = origin->asString()->clone();
            else
               (*array)[ pos ] = *origin;
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_RANGE:
      {
         CoreArray *array =  operand1->asArray();
         register int32 end = operand2->asRangeIsOpen() ? array->length() : operand2->asRangeEnd();
         register int32 start = operand2->asRangeStart();
         if( origin->type() == FLC_ITEM_ARRAY ) {
            if( array->change( *origin->asArray(), start, end ) )
               return;
         }
         else {
            if ( start != end ) {// insert
               if( ! array->remove( start, end ) )
                  break;
            }
            if ( origin->isString() && origin->asString()->garbageable() ) {
               if( array->insert( origin->asString()->clone(), start ) )
                  return;
            }
            else {
               if( array->insert( *origin, start ) )
                  return;
            }
         }
      }
      break;

      case FLC_ITEM_RANGE << 8 | FLC_ITEM_INT:
      case FLC_ITEM_RANGE << 8 | FLC_ITEM_NUM:
      {
          int32 pos = (int32) operand2->forceInteger();
         if ( origin->isOrdinal() )
         {

            switch( pos )
            {
               case 0:
                  operand1->setRange( (int32) origin->forceInteger(),
                        (int32) operand1->asRangeEnd(),
                        operand1->asRangeIsOpen() );
               return;

               case 1: case -1:
                  operand1->setRange( operand1->asRangeStart(),
                        (int32) origin->forceInteger(),
                        false );
               return;
            }
         }
         else if ( origin->isNil() && ( pos == -1 || pos == 1 ) )
         {
            operand1->setRange( operand1->asRangeStart(),
                  0,
                  true );
            return;
         }

      }
      break;
   }

    vm->raiseRTError(
               new RangeError( ErrorParam( e_arracc ).origin( e_orig_vm ) ) );
}

//51
void opcodeHandler_STP( register VMachine *vm )
{
   Item *target = vm->getOpcodeParam( 1 )->dereference();
   Item *method = vm->getOpcodeParam( 2 )->dereference();
   Item *source = vm->getOpcodeParam( 3 )->dereference();

   if ( method->isString() && target->isObject() )
   {
      Item temp;

      // Are we restoring an original item?
      if( source->isMethod() && source->asMethodObject() == target->asObject() )
      {
         temp.setFunction( source->asMethodFunction(), source->asModule() );
         source = &temp;
      }
      else if ( source->isString() && source->asString()->garbageable() )
      {
         GarbageString *gcs = new GarbageString( vm, *source->asString() );
         source->setString( gcs );
      }

      if( target->asObject()->setProperty( *method->asString(), *source ) )
            return;

      vm->raiseRTError(
         new RangeError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
            extra( *method->asString() ) ) );

      return;
   }

   vm->raiseError( e_prop_acc, "STP" );
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
   // do not decode the first two operands.
   opcodeHandler_LDV( vm );
   // Luckily, we can decode the third operand here, without any risk, as LDV never uses m_pc_next
   Item *operand3 = vm->getOpcodeParam( 3 )->dereference();

   if( operand3 != &vm->m_regA )
      operand3->copy( vm->m_regA );
}

//53
void opcodeHandler_LDPT( register VMachine *vm )
{
   // do not decode the first two operands.
   opcodeHandler_LDP( vm );
   // Luckily, we can decode the third operand here, without any risk, as LDV never uses m_pc_next
   Item *operand3 = vm->getOpcodeParam( 3 )->dereference();

   if( operand3 != &vm->m_regA )
      operand3->copy( vm->m_regA );
}

//54
void opcodeHandler_STVR( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();
   // do not deref op3
   Item *origin = vm->getOpcodeParam( 3 );

   GarbageItem *gitem;
   if( ! origin->isReference() )
   {
      gitem = new GarbageItem( vm, *origin );
      origin->setReference( gitem );
   }

   // try to access a dictionary with every item
   // access addition.
   if( operand1->type() == FLC_ITEM_DICT ) {
      operand1->asDict()->insert( *(operand2), *origin );
      return;
   }

   switch( operand1->type() << 8 | operand2->type() )
   {
      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_INT:
      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_NUM:
      {
         register int32 pos = (int32) operand2->forceInteger();
         CoreArray *array = operand1->asArray();
         if ( pos >= (-(int)array->length()) && pos < (int32) array->length() ) {
            (*array)[ pos ] = *origin;
            return;
         }
      }
      break;

      case FLC_ITEM_ARRAY << 8 | FLC_ITEM_RANGE:
      {
         CoreArray *array =  operand1->asArray();
         register int32 end = operand2->asRangeIsOpen() ? array->length() : operand2->asRangeEnd();
         register int32 start = operand2->asRangeStart();
         if ( start != end ) {// insert
            if( ! array->remove( start, end ) )
               break;
         }
         if( array->insert( *origin, start ) )
            return;
      }
      break;
   }

    vm->raiseRTError(
            new RangeError( ErrorParam( e_arracc ).origin( e_orig_vm ).extra("STVR") ) );
}

//55
void opcodeHandler_STPR( register VMachine *vm )
{
   Item *target = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();
   // do not deref op3
   Item *source = vm->getOpcodeParam( 3 );


   switch( target->type() )
   {
      case FLC_ITEM_OBJECT:
         if ( operand2->type() == FLC_ITEM_STRING ) {
            // If the source is a method, we have to put in just its function, discarding the recored object.
            if( source->type() == FLC_ITEM_METHOD ) {
               // will instantiate by value
               Item temp( source->asMethodFunction(), source->asModule() );
               if( target->asObject()->setProperty( *operand2->asString(), temp ) )
                  return;
            }
            else {
               if( ! source->isReference() ) {
                  GarbageItem *gitem = new GarbageItem( vm, *source );
                  source->setReference( gitem );
               }

               if( target->asObject()->setProperty( *operand2->asString(), *source ) )
                  return;
            }

            vm->raiseRTError(
               new RangeError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
                  extra( *operand2->asString() ) ) );
            return;

         }
      break;

      // classes cannot have their properties set (for now).
   }

   vm->raiseRTError(
               new RangeError( ErrorParam( e_prop_acc ).origin( e_orig_vm ).extra("STPR") ) );

}

//56
void opcodeHandler_TRAV( register VMachine *vm )
{
   // we need some spare space. We preallocate it because if the parameters are in
   // the stack, we need their pointers to stay valid.
   vm->m_stack->resize( vm->m_stack->size() + 3 );

   // get the jump label.
   int wayout = vm->getNextNTD32();
   Item *real_dest = vm->getOpcodeParam( 2 );  // we may need it undereferenced
   Item *source = vm->getOpcodeParam( 3 )->dereference();

   Item *dest = real_dest->dereference();

   switch( source->type() )
   {
      case FLC_ITEM_ARRAY:
      {
         CoreArray *array = source->asArray();
         if( array->length() == 0 ) {
            goto trav_go_away;
         }

         Item *sourceItem = &(*array)[ 0 ];
         // is dest a distributed array? -- it has been pushed in the stack
         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            uint64 varCount = dest->asInteger();

            if( sourceItem->type() != FLC_ITEM_ARRAY ||
               varCount != sourceItem->asArray()->length() )
            {
               vm->raiseRTError(
                  new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
               return;
            }

            for ( uint32 i = 0; i < varCount; i ++ ) {
               vm->stackItem( vm->m_stack->size() - (uint32)varCount + i - 3).dereference()->copy(
                        * ((* sourceItem->asArray() )[i].dereference()) );
            }
         }
         else
            dest->copy( *(sourceItem->dereference()) );

         // prepare ... iterator
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = ( (int64) 0 );
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *dict = source->asDict();
         if( dict->length() == 0 ) {
            goto trav_go_away;
         }

         if( ! ( vm->operandType( 1 ) == P_PARAM_INT32 ||  vm->operandType( 1 ) == P_PARAM_INT64 ) ||
              dest->asInteger() != 2 )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }

         // we need an iterator...
         DictIterator *iter = dict->first();

         // in a dummy object...
         CoreObject *obj = new CoreObject( vm, iter );

         register int stackSize = vm->m_stack->size();
         vm->stackItem( stackSize - 5 ).dereference()->
               copy( iter->getCurrentKey() );
         vm->stackItem( stackSize - 4 ).dereference()->
               copy( *iter->getCurrent().dereference() );

         // prepare... iterator
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = obj;
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *obj = source->asObject();
         if( obj->getUserData() == 0 || ! obj->getUserData()->isSequence() )
         {
            vm->raiseError( e_invop, "TRAV" );
            return;
         }

         Sequence *seq = static_cast< Sequence *>( obj->getUserData() );
         if ( seq->empty() )
         {
            goto trav_go_away;
         }

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }

         // we need an iterator...
         CoreIterator *iter = seq->getIterator();

         // in a dummy object...
         obj = new CoreObject( vm, iter );

         *dest->dereference() = iter->getCurrent();

         // prepare... iterator
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = obj;
      }
      break;

      case FLC_ITEM_ATTRIBUTE:
      {
         Attribute *attrib = source->asAttribute();
         if ( attrib->empty() )
         {
            goto trav_go_away;
         }

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }

         // we need an iterator...
         AttribIterator *iter = attrib->getIterator();

         // in a dummy object...
         CoreObject *obj = new CoreObject( vm, iter );

         *dest->dereference() = iter->getCurrent();

         // prepare... iterator
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = obj;
      }
      break;

      case FLC_ITEM_STRING:
         if( source->asString()->length() == 0 )
            goto trav_go_away;

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }
         *dest->dereference() = new GarbageString( vm, source->asString()->subString(0,1) );
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = ( (int64) 0 );
         break;

      case FLC_ITEM_MEMBUF:
         if( source->asMemBuf()->length() == 0 )
            goto trav_go_away;

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }
         *dest->dereference() = (int64) source->asMemBuf()->get(0);
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = ( (int64) 0 );
         break;

      case FLC_ITEM_RANGE:
         if( ! source->asRangeIsOpen() && source->asRangeEnd() == source->asRangeStart() )
         {
            goto trav_go_away;
         }

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }
         *dest->dereference() = (int64) source->asRangeStart();
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = ( (int64) source->asRangeStart() );
         break;

      case FLC_ITEM_NIL:
         // jump out
         goto trav_go_away;


      default:
         vm->raiseError( e_invop, "TRAV" );
         goto trav_go_away;
   }

   // after the iterator/counter, push the source
   if ( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
   {
      vm->m_stack->itemAt( vm->m_stack->size()-2 ) = dest->asInteger();
   }
   else {
      Item refDest;
      vm->referenceItem( refDest, *real_dest );
      vm->m_stack->itemAt( vm->m_stack->size()-2 ) = refDest;
   }

   // and then the source by copy
   vm->m_stack->itemAt( vm->m_stack->size()-1 ) = *source;

   // we're done.
   return;

trav_go_away:
   vm->m_pc_next = wayout;
   uint32 vars = 0;
   // eventually pop referenced vars
   if ( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
   {
      vars = (uint32) dest->asInteger();
   }

   if( vars + 3 > vm->m_stack->size() )
   {
      vm->raiseError( e_stackuf, "TRAV" );
   }
   else {
      vm->m_stack->resize( vm->m_stack->size() - vars - 3 );
   }
}


//57
void opcodeHandler_FORI( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();
   Item *operand3 = vm->getOpcodeParam( 3 )->dereference();

   // get the third operand.
   register int size = vm->m_stack->size();
   if ( size < 2 )  {
      vm->raiseError( e_stackuf, "FORI" );
      return;
   }

   // check parameters
   if ( ! vm->m_stack->itemAt( size - 1 ).isScalar() )
   {
      vm->raiseRTError(
               new TypeError( ErrorParam( e_for_user_error ).origin( e_orig_vm ).extra("FORI step") ) );
      return;
   }

   if ( ! vm->m_stack->itemAt( size - 2 ).isScalar() )
   {
      vm->raiseRTError(
               new TypeError( ErrorParam( e_for_user_error ).origin( e_orig_vm ).extra("FORI target") ) );
      return;
   }

   if ( ! operand3->isScalar() )
   {
      vm->raiseRTError(
               new TypeError( ErrorParam( e_for_user_error ).origin( e_orig_vm ).extra("FORI start") ) );
      return;
   }

   // check step
   int64 target = vm->m_stack->itemAt( size - 2 ).forceInteger();
   int64 step = vm->m_stack->itemAt( size - 1 ).forceInteger();
   int64 begin =  operand3->forceInteger();

   if( step == 0 )
   {
      // define step
      step = begin > target ? -1:1;
      vm->m_stack->itemAt( size - 1 ) = step;
   }
   else {
      if ( step > 0 && begin > target || step < 0 && begin < target ) {
         // immediate exit
         vm->m_stack->resize( size - 2 );
         vm->m_pc_next = pNext;
         return;
      }
   }

   // initialize the counter
   operand2->setInteger( begin );
}


//58
void opcodeHandler_FORN( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   // get the third operand.
   register int size = vm->m_stack->size();
   if ( size < 2 )  {
      vm->raiseError( e_stackuf, "FORN" );
      return;
   }
   // let's suppose the most important controls have been performed by forn
   if ( ! operand2->isScalar() )
   {
      vm->raiseRTError(
               new TypeError( ErrorParam( e_for_user_error ).origin( e_orig_vm ).extra("FORN variable") ) );
      return;
   }

   int64 target = vm->m_stack->itemAt( size - 2 ).forceInteger();
   int64 step = vm->m_stack->itemAt( size - 1 ).forceInteger();
   int64 current =  operand2->forceInteger();
   current += step;

   if ( step > 0 && current > target || step < 0 && current < target ) {
      // end of loop
      vm->m_stack->resize( size - 2 );
   }
   else {
      operand2->setInteger( current );
      vm->m_pc_next = pNext;
   }
}

//59
void opcodeHandler_SHL( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      vm->m_regA.setInteger( operand1->asInteger() << operand2->asInteger() );
   else
      vm->raiseRTError(
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHL") ) );

}

//5A
void opcodeHandler_SHR( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      vm->m_regA.setInteger( operand1->asInteger() >> operand2->asInteger() );
   else
      vm->raiseRTError(
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHR") ) );
}

//5B
void opcodeHandler_SHLS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      operand1->setInteger( operand1->asInteger() << operand2->asInteger() );
   else
      vm->raiseRTError(
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHLS") ) );
}

//5C
void opcodeHandler_SHRS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if( operand1->isInteger() && operand2->isInteger() )
      operand1->setInteger( operand1->asInteger() >> operand2->asInteger() );
   else
      vm->raiseRTError(
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra("SHRS") ) );
}

//5E
void opcodeHandler_LDVR( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   if ( operand1->isArray() )
   {
      if ( operand2->isOrdinal() )
      {
         register int32 pos = (int32) operand2->forceInteger();
         CoreArray &array = *operand1->asArray();
         if ( -pos > (int32)array.length() || pos >= (int32)array.length() )
             vm->raiseRTError(
               new RangeError( ErrorParam( e_arracc ).origin( e_orig_vm ) ) );
         else {
            GarbageItem *gitem;
            Item *ref = array.elements() + pos;
            if ( ref->isReference() ) {
               gitem = ref->asReference();
            }
            else{
               gitem= new GarbageItem( vm, *ref );
               ref->setReference( gitem );
            }
            vm->m_regA.setReference( gitem );
         }
         return;
      }
   }

   // try to access a dictionary with every item
   else if( operand1->type() == FLC_ITEM_DICT )
   {
      CoreDict &dict =  *operand1->asDict();
      Item *value;
      if( ( value = dict.find( *operand2 ) ) != 0  )
      {
         GarbageItem *gitem;
         // we don't use memPool::referenceItem for performace reasons
         if ( value->isReference() ) {
            gitem = value->asReference();
         }
         else{
            gitem = new GarbageItem( vm, *value );
            value->setReference( gitem );
         }
         vm->m_regA.setReference( gitem );
         return;
      }
   }

   vm->raiseRTError(
               new RangeError( ErrorParam( e_arracc ).origin( e_orig_vm ).extra( "LDVR" ) ) );
}

//5F
void opcodeHandler_LDPR( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   Item *source = operand1;
   CoreObject *self = vm->m_regS1.type() == FLC_ITEM_NIL? 0: vm->m_regS1.asObject();
   CoreClass *sourceClass=0;

   switch( source->type() )
   {
      case FLC_ITEM_OBJECT:
         if ( operand2->type() == FLC_ITEM_STRING )
         {
            Item prop;
            if( source->asObject()->getProperty( *operand2->asString(), prop ) )
            {
               // we must create a method if the property is a function.
               GarbageItem *gitem;
               // we don't use memPool::referenceItem for performace reasons
               if ( prop.isReference() ) {
                  gitem = prop.asReference();
               }
               else{
                  gitem = new GarbageItem( vm, prop );
                  prop.setReference( gitem );
                  source->asObject()->setProperty( *operand2->asString(), prop );
               }
               vm->m_regA.setReference( gitem );
               return;
            }

            vm->raiseRTError(
               new RangeError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
                  extra( *operand2->asString() ) ) );
            return;
         }
      break;
   }

   vm->raiseRTError(
               new RangeError( ErrorParam( e_prop_acc ).origin( e_orig_vm ).extra( "LDPR" ) ) );
}

// 60
void opcodeHandler_POWS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   numeric powval;

   errno = 0;
   switch( operand2->type() )
   {
      case FLC_ITEM_INT:
      {
         numeric val2 = (numeric) operand2->asInteger();
         if ( val2 == 0 ) {
            operand1->setInteger( (int64) 1 );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               powval = ::pow( (numeric) operand1->asInteger(), val2 );
            break;

            case FLC_ITEM_NUM:
               powval = ::pow( operand1->asNumeric(), val2 );
            break;

            default:
               vm->raiseRTError(
                  new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra( "POWS" ) ) );
               return;
         }
      }
      break;

      case FLC_ITEM_NUM:
      {
         numeric val2 = operand2->asNumeric();
         if ( val2 == 0.0 ) {
            operand1->setInteger( (int64) 1 );
            return;
         }

         switch( operand1->type() ) {
            case FLC_ITEM_INT:
               powval = pow( (double) operand1->asInteger(), val2 );
            break;

            case FLC_ITEM_NUM:
               powval = pow( operand1->asNumeric(), val2 );
            break;

            default:
               vm->raiseRTError(
                  new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra( "POWS" ) ) );
               return;
         }
      }
      break;

      default:
         vm->raiseRTError(
            new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra( "POWS" ) ) );
         return;
   }

   if ( errno != 0 )
   {
      vm->raiseRTError( new MathError( ErrorParam( e_domain ).origin( e_orig_vm ) ) );
   }
   else {
      operand1->setNumeric( powval );
   }
}

//61
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
            vm->raiseRTError(
               new RangeError( ErrorParam( e_arracc ).origin( e_orig_vm ) ) );
      }
      return;
   }

   vm->raiseRTError(
      new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra( "LSB" ) ) );
}

//62
void opcodeHandler_UNPS( register VMachine *vm )
{
   uint32 len = (uint32) vm->getNextNTD32();
   Item *operand2 = vm->getOpcodeParam( 2 )->dereference();

   uint32 base = vm->m_stack->size();

	if ( len > base )
	{
      vm->raiseError( e_stackuf, "UNPS" );
      return;
	}

   base -= len;

   if ( operand2->type() != FLC_ITEM_ARRAY ) {
      vm->m_stack->resize( base );
      vm->raiseRTError(
         new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra( "UNPS" ) ) );
      return;
   }

   CoreArray *source = operand2->asArray();

   Item *s_elems = source->elements();
	if ( len != source->length() )
	{
      vm->raiseRTError(
         new RangeError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "UNPS" ) ) );
      return;
   }

   for ( uint32 i = 0; i < len; i++ )  {
      *vm->stackItem(base + i).dereference() = *s_elems[i].dereference();
   }

   vm->m_stack->resize( base );
}

//61
void opcodeHandler_SELE( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *real_op2 = vm->getOpcodeParam( 2 );
   Item *op2 = real_op2->dereference();
   uint64 sw_count = (uint64) vm->getNextNTD64();

   byte *tableBase = vm->m_code + vm->m_pc_next;

   // SELE B has a special semantic used as a TRY/CATCH helper
   bool hasDefault = endianInt32( *reinterpret_cast<uint32 *>( tableBase )  ) == 0;
   bool reRaise = real_op2 == &vm->regB();

   // In select we use only integers and object fields.
   uint16 sw_int = (int16) (sw_count >> 48);
   uint16 sw_obj = (int16) sw_count;

   //determine the value type to be checked
   int type = op2->type();
   if ( sw_int > 0 )
   {
      if ( vm->seekInteger( type, tableBase + sizeof(uint32), sw_int, vm->m_pc_next ) )
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
                  vm->m_pc_next ) )
               return;
   }

   if ( reRaise && hasDefault ) // in case of re-raise
   {
      vm->m_event = VMachine::eventRisen;
      // error already in B, as we didn't execute any code.
      return;
   }

   // in case of failure go to default or past sele...
   vm->m_pc_next = pNext;
}

//62
void opcodeHandler_INDI( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( ! operand1->isString() )
   {
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("INDI").origin( e_orig_vm ) ) );
      return;
   }

   Item *itm = vm->findLocalVariable( *operand1->asString() );
   if ( itm == 0 )
   {
       vm->raiseRTError(
            new ParamError( ErrorParam( e_param_indir_code ).origin( e_orig_vm ).extra( "INDI" ) ) );
   }
   else {
      vm->m_regA = *itm;
   }
}

//63
void opcodeHandler_STEX( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( ! operand1->isString() )
   {
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("STEX").origin( e_orig_vm ) ) );
      return;
   }

   GarbageString *target = new GarbageString( vm );
   String *src = operand1->asString();

   VMachine::returnCode retval = vm->expandString( *src, *target );
   switch( retval )
   {
      case VMachine::return_ok:
         vm->m_regA = target;
      break;

      case VMachine::return_error_parse_fmt:
         vm->raiseRTError(
            new TypeError( ErrorParam( e_fmt_convert ).origin( e_orig_vm ).extra( "STEX" ) ) );
      break;

      case VMachine::return_error_string:
         vm->raiseRTError(
            new ParamError( ErrorParam( e_param_strexp_code ).origin( e_orig_vm ).extra( "STEX" ) ) );
      break;

      case VMachine::return_error_parse:
         vm->raiseRTError(
               new ParamError( ErrorParam( e_param_indir_code ).origin( e_orig_vm ).extra( "STEX" ) ) );
      break;
   }
}

//64
void opcodeHandler_TRAC( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( vm->m_stack->size() < 3 ) {
      vm->raiseError( e_stackuf, "TRAC" );
      return;
   }

   register int size = vm->m_stack->size();
   Item *iterator = &vm->stackItem( size - 3 );
   bool isIterator = ! iterator->isInteger();
   Item *source = &vm->stackItem( size - 1 );

   Item *copied = operand1;

   Item *dest = 0;

   switch( source->type() )
   {
      case FLC_ITEM_ARRAY:
      {
         if ( isIterator )
         {
            vm->raiseError( e_stackuf, "TRAC" );
            return;
         }

         CoreArray *sarr = source->asArray();
         uint32 counter = (uint32) iterator->asInteger();

         if ( counter >= sarr->length() ) {
            // item has been killed. We should do nothing but exit.
            return;
         }

         dest = sarr->at( counter ).dereference();

      }
      break;

      case FLC_ITEM_DICT:
      case FLC_ITEM_OBJECT:
      case FLC_ITEM_ATTRIBUTE:
         if ( ! isIterator )
         {
            vm->raiseError( e_stackuf, "TRAC" );
            return;
         }
         else {
            CoreIterator *iter = (CoreIterator *) iterator->asObject()->getUserData();

            if( ! iter->isValid() )
            {
               // item has been killed. We should do nothing but exit.
               return;
            }

            dest = iter->getCurrent().dereference();
         }
      break;

      case FLC_ITEM_MEMBUF:
         if ( isIterator )
         {
            vm->raiseError( e_stackuf, "TRAC" );
         }
         else {
            uint32 counter = (uint32) iterator->asInteger();
            MemBuf *mb = source->asMemBuf();
            if ( mb->length() < counter )
            {
               return;
            }

            if( copied->isOrdinal() )
               mb->set( counter, (uint32) copied->forceInteger() );
            else {
               vm->raiseError( e_invop, "TRAC" );
            }
         }
      // always return, we've managed the string.
      return;

      case FLC_ITEM_STRING:
         if ( isIterator )
         {
            vm->raiseError( e_stackuf, "TRAC" );
         }
         else {
            uint32 counter = (uint32) iterator->asInteger();
            String *sstr = source->asString();
            if ( sstr->length() < counter )
            {
               return;
            }

            if( copied->isString() )
            {
                  sstr->change( counter, counter + 1, *copied->asString() );
            }
            else if( copied->isOrdinal() )
               sstr->setCharAt( counter, (uint32) copied->forceInteger() );
            else {
               vm->raiseError( e_invop, "TRAC" );
            }
         }
      // always return, we've managed the string.
      return;

      case FLC_ITEM_RANGE:
         // we don't need anything here.
         return;

      default:
         vm->raiseError( e_invop, "TRAC" );
         return;
   }

   if( copied->isString() )
      *dest = new GarbageString( vm, *copied->asString() );
   else
      *dest = *copied;

}

//65
void opcodeHandler_WRT( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   Stream *stream = vm->stdOut();
   if ( stream == 0 )
   {
      vm->raiseError( e_invop, "WRT" );
      return;
   }

   if( operand1->isString() )
   {
      stream->writeString( *operand1->asString() );
   }
   else {
      String temp;
      vm->itemToString( temp, operand1 );
      if( ! vm->hadError() )
      {
         stream->writeString( temp );
      }
   }
}

}


/* end of vm_run.cpp */
