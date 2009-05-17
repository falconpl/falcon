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
#include <falcon/cdict.h>
#include <falcon/corefunc.h>
#include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/membuf.h>
#include <falcon/vmmsg.h>

#include <math.h>
#include <errno.h>
#include <string.h>

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
         m_imm[bc_pos].setInteger( grabInt64( m_code + m_pc_next  ) );
         m_pc_next += sizeof( int64 );
      return m_imm + bc_pos;

      case P_PARAM_STRID:
         {
            String *temp = currentLiveModule()->getString( endianInt32(*reinterpret_cast<int32 *>( m_code + m_pc_next ) ) );
            //m_imm[bc_pos].setString( temp, const_cast<LiveModule*>(currentLiveModule()) );
            m_imm[bc_pos].setString( temp );
            m_pc_next += sizeof( int32 );
         }
      return m_imm + bc_pos;

      case P_PARAM_LBIND:
         m_imm[bc_pos].setLBind( currentLiveModule()->getString(
            endianInt32(*reinterpret_cast<int32 *>( m_code + m_pc_next ) ) ) );
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
      case P_PARAM_REGL1: return &m_regL1;
      case P_PARAM_REGL2: return &m_regL2;
   }

   // we should not be here.
   fassert( false );
   return 0;
}


void VMachine::run()
{
   tOpcodeHandler *ops = m_opHandlers;
   m_event = eventNone;
   Error *the_error = 0;

   // declare this as the running machine
   setCurrent();

   /*
   class AutoIdle
   {
      VMachine  *m_vm;
   public:
      inline AutoIdle( VMachine *vm ):
         m_vm(vm) { vm->pulseIdle(); }
         // ^^we are already idle, but pulseidle force to honor pending
         // blocking requets.
      inline ~AutoIdle() { m_vm->idle(); }
   } l_autoidle(this);
   */

   while( 1 )
   {
      // move m_pc_next to the end of instruction and beginning of parameters.
      // in case of opcode in the call_request range, this will move next pc to return.
      m_pc_next = m_pc + sizeof( uint32 );

      // external call required?
      if ( m_pc >= i_pc_call_request )
      {
         try {
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
                  m_regA.setNil();
                  m_symbol->getExtFuncDef()->call( this );
            }
         }
         catch( Error *err )
         {
            // fake an error raisal
            the_error = err;
            m_event = eventRisen;
         }
      }
      else
      {
         // execute the opcode.
         try {
            ops[ m_code[ m_pc ] ]( this );
         }
         catch( Error *err )
         {
            m_event = eventRisen;
            the_error = err;
            err->origin( e_orig_vm );
         }
      }

      m_opCount ++;

      //=========================
      // Executes periodic checks

      if ( m_opCount > m_opNextCheck )
      {
         // By default, if nothing else happens, we should do a check no sooner than this.
         m_opNextCheck = m_opCount + FALCON_VM_DFAULT_CHECK_LOOPS;

         // pulse VM idle
         if( m_bGcEnabled )
            m_baton.checkBlock();

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

            // perform messages
            m_mtx_mesasges.lock();
            while( m_msg_head != 0 )
            {
               VMMessage* msg = m_msg_head;
               m_msg_head = msg->next();
               // it is ok if m_msg_tail is left dangling.
               m_mtx_mesasges.unlock();

               processMessage( msg );
               //delete msg;

               // see if we have more messages in the meanwhile
               m_mtx_mesasges.lock();
            }

            m_msg_tail = 0;
            m_mtx_mesasges.unlock();
         }
      }

      //===============================
      // consider requests.
      //
      switch( m_event )
      {

         // This events leads to VM main loop exit.
         case eventInterrupt:
            if ( m_atomicMode )
            {
               the_error = new InterruptedError( ErrorParam( e_interrupted ).origin( e_orig_vm ).
                     symbol( m_symbol->name() ).
                     module( m_currentModule->module()->name() ).
                     line( __LINE__ ).
                     hard() );
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

            if( m_tryFrame == i_noTryFrame && the_error == 0 )  // uncaught error raised from scripts...
            {
               // create the error that the external application will see.
               Error *err;
               if ( m_regB.isOfClass( "Error" ) )
               {
                  // in case of an error of class Error, we have already a good error inside of it.
                  err = static_cast<core::ErrorObject *>(m_regB.asObjectSafe())->getError();
                  err->incref();
               }
               else {
                  // else incapsulate the item in an error.
                  err = new GenericError( ErrorParam( e_uncaught ).origin( e_orig_vm ) );
                  err->raised( m_regB );
               }
               the_error = err;
            }

            if( the_error != 0 && ! the_error->hasTraceback() )
               fillErrorContext( the_error, true );

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
                  {
                     if ( the_error != 0 )
                     {
                        throw the_error;
                     }
                     return;
                  }
               }
               // call return may raise eventQuit, but only when m_stackBase is zero,
               // so we don't consider it.
            }

            // We are in the frame that should handle the error, in one way or another
            // should we catch it?
            // If the error is zero, we know we have a script exception ready to be caught
            // as we have filtered it before
            if ( the_error == 0 )
            {
               popTry( true );
               m_event = eventNone;
               continue;
            }
            // else catch it only if allowed.
            else if( the_error->catchable() && m_tryFrame != i_noTryFrame )
            {
               CoreObject *obj = the_error->scriptize( this );
               the_error->decref();
               the_error = 0;
               if ( obj != 0 )
               {
                  // we'll manage the error throuhg the obj, so we release the ref.
                  m_regB.setObject( obj );
                  popTry( true );
                  //the_error->decref();  // scriptize adds a reference
                  m_event = eventNone;
                  continue;
               }
               else {
                  // Panic. Should not happen -- scriptize has raised a symbol not found error
                  // describing the missing error class; we must tell the user so that the module
                  // not declaring the correct error class, or failing to export it, can be
                  // fixed.
                  throw the_error;
               }
            }
            // we couldn't catch the error (this also means we're at m_stackBase zero)
            // we should handle it then exit
            else {
               // we should manage the error; if we're here, m_stackBase is zero,
               // so we are the last in charge
               throw the_error;
            }
         break;

         case eventYield:
            m_pc = m_pc_next;
            m_event = eventNone;
            try
            {
               yield( m_yieldTime );
               if ( m_event == eventSleep )
                  return;
               m_event = eventNone;
            }
            catch( Error* e )
            {
               m_pc = i_pc_redo_request;
               the_error = e;
               m_event = eventRisen;
            }
            
         continue;

         // this can only be generated by an electContext or rotateContext that finds there's the need to sleep
         // as contexts cannot be elected in atomic mode, and as wait event is
         // already guarded against atomic mode breaks, we let this through
         case eventSleep:
            return;

         case eventWait:
            if ( m_atomicMode )
            {
               the_error = new InterruptedError( ErrorParam( e_interrupted ).origin( e_orig_vm ).
                     symbol( "vm_run" ).
                     module( "core.vm" ).
                     line( __LINE__ ).
                     hard() );
               // just re-parse the event
               m_pc = i_pc_redo_request;
               m_event = eventRisen;
               continue;
            }
            if ( m_sleepingContexts.empty() && !m_sleepAsRequests && m_yieldTime < 0.0 )
            {
               Error* the_error = new GenericError( ErrorParam( e_deadlock ).origin( e_orig_vm ) );
               fillErrorContext( the_error );
               m_event = eventRisen;
               throw the_error;
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
   CoreArray *array = new CoreArray( size );

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
   LinearDict *dict = new LinearDict( length );

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
   if ( data->isFutureBind() )
   {
      // with this marker, the next call operation will search its parameters.
      // Let's consider this a temporary (but legitimate) hack.
      vm->m_regBind.flags( 0xF0 );
   }
   vm->m_stack->push( data );
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
      {
         CoreIterator *iter = (CoreIterator *) iterator->asGCPointer();
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
      {
         if ( source->asRangeIsOpen() )
         {
            vm->m_pc_next = pcNext;
         }

         int64 increment = source->asRangeStep();
         if ( source->asRangeStart() < source->asRangeEnd() )
         {
            if ( increment == 0 )
               increment = 1;

            if ( iterator->asInteger() + increment >= source->asRangeEnd() )
               vm->m_pc_next = pcNext;
         }
         else {
            if ( increment == 0 )
               increment = -1;

            if ( iterator->asInteger() + increment < source->asRangeEnd() )
               vm->m_pc_next = pcNext;
         }
      }
      break;

      default:
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("TRAL").origin( e_orig_vm ) ) );
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
      vm->raiseRTError( new TypeError( ErrorParam( e_bitwise_op ).extra("BNOT").origin( e_orig_vm ) ) );
}

//1B
void opcodeHandler_NOTS( register VMachine *vm )
{
   register Item *operand1 = vm->getOpcodeParam( 1 )->dereference();

   if ( operand1->isOrdinal() )
      operand1->setInteger( ~operand1->forceInteger() );
   else
      vm->raiseRTError( new TypeError( ErrorParam( e_bitwise_op ).extra("NOTS").origin( e_orig_vm ) ) );
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

   // create the coroutine
   vm->coPrepare( pSize );

   // fork
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
      operand1->setString( new CoreString( *operand2->asString() ) );
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
   Item *operand3 =  vm->getOpcodeParam( 3 )->dereference();


   int64 secondOp = operand2->forceIntegerEx();
   if( operand2->isOob() && secondOp > 0 )
      secondOp++;

   int64 step;
   if ( operand3->isNil() )
   {
      step = vm->m_stack->itemAt( vm->m_stack->size() - 1).forceIntegerEx();
      vm->m_stack->pop();
   }
   else {
      step = operand3->forceIntegerEx();
   }

   int64 firstOp = operand1->forceIntegerEx();
   if( step == 0 )
   {
      if ( firstOp <= secondOp )
          step = 1;
      else
         step = -1;
   }

   vm->m_regA.setRange( new CoreRange(
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

   operand2->readyFrame( vm, pNext );
}

//3B
void opcodeHandler_INST( register VMachine *vm )
{
   uint32 pNext = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();

   if( operand2->type() != FLC_ITEM_CLASS )
   {
      vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("INST").origin( e_orig_vm ) ) );
      return;
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
   else if ( operand2->isMethod() )
   {
      call = operand2->asMethodFunc()->symbol();
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
      vm->raiseRTError(
         new AccessError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
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
                  new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
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
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
            return;
         }
         else {
            DictIterator *iter = (DictIterator *) iterator->asGCPointer();

            if( ! iter->isValid() )
            {
               vm->raiseError( e_arracc, "TRAN" );
               return;
            }

            if( p3 == 1 )
            {
               CoreDict *sdict = source->asDict();
               sdict->remove( *iter );
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

            vm->stackItem( size -3 - 2 ).dereference()->
                  copy( iter->getCurrentKey() );
            vm->stackItem( size -3 - 1 ).dereference()->
                  copy( *iter->getCurrent().dereference() );
         }
      break;

      case FLC_ITEM_OBJECT:
         if ( ! isIterator )
         {
            vm->raiseError( e_arracc, "TRAN" );
            return;
         }
         else {
            CoreIterator *iter = (CoreIterator *) iterator->asGCPointer();

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
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
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

            *dest->dereference() = new CoreString(
                  sstr->subString( counter, counter + 1 ) );
         }
      break;

      case FLC_ITEM_MEMBUF:
         if ( isIterator || isDestStack )
         {
            vm->raiseRTError(
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
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
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAN" ) ) );
            return;
         }
         else {
            int64 counter = iterator->asInteger();
            int64 increment = source->asRangeStep();
            // if( p3 == 1 ) -- we Ignore this case, and let continue dropping to act as continue.

            if( source->asRangeStart() < source->asRangeEnd() )
            {
               counter += increment == 0 ? 1 : increment;
               if ( counter >= source->asRangeEnd() ) {
                  vm->m_pc_next = p2;
                  return;
               }
            }
            else {
               counter += increment == 0 ? -1 : increment;
               if ( counter < source->asRangeEnd() ) {
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
void opcodeHandler_LDAS( register VMachine *vm )
{
   uint32 size = (uint32) vm->getNextNTD32();
   Item *operand2 =  vm->getOpcodeParam( 2 )->dereference();
   
   if( operand2->isArray() )
   {
      CoreArray* arr = operand2->asArray();
      if( arr->length() != size )
      {
         vm->raiseRTError(
            new AccessError( ErrorParam( e_unpack_size )
               .extra( "LDAS" )
               .origin( e_orig_vm ) ) );
      }
      else {
         uint32 oldpos = vm->m_stack->size();
         vm->m_stack->resize( oldpos + size );
         void* mem = vm->m_stack->itemPtrAt( oldpos );
         memcpy( mem, arr->elements(), size * sizeof(Item) );
     }
   }
   else {
      vm->raiseRTError(
            new AccessError( ErrorParam( e_invop )
               .extra( "LDAS" )
               .origin( e_orig_vm ) ) );
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
   if(  vm->m_stack->empty() )
   {
      vm->raiseError( e_stackuf, "STVS" );
      return;
   }

   Item *operand1 =  vm->getOpcodeParam( 1 );
   Item *operand2 =  vm->getOpcodeParam( 2 );
   Item *origin = &vm->m_stack->topItem();

   operand1->setIndex( *operand2, *origin );
   vm->regA() = *origin;
   vm->m_stack->pop();

}

//4A
void opcodeHandler_STPS( register VMachine *vm )
{
   if( vm->m_stack->empty() )
   {
      vm->raiseError( e_stackuf, "STPS" );
      return;
   }

   Item *target =  vm->getOpcodeParam( 1 );
   Item *method = vm->getOpcodeParam( 2 )->dereference();

   if ( method->isString() )
   {
      target->setProperty( *method->asString(), vm->m_stack->topItem() );
      vm->regA() = vm->m_stack->topItem();
      vm->m_stack->pop();
   }
   else
   {
      vm->m_stack->pop();
      vm->raiseRTError( new TypeError( ErrorParam( e_prop_acc ).extra("STPS").origin( e_orig_vm ) ) );
   }
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

   vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("STP").origin( e_orig_vm ) ) );

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
      vm->raiseRTError(
         new AccessError( ErrorParam( e_prop_acc ).origin( e_orig_vm ) .
            extra( operand2->isString() ? *operand2->asString() : "?" ) ) );
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

   vm->raiseRTError( new TypeError( ErrorParam( e_prop_acc ).extra("STPR").origin( e_orig_vm ) ) );
   return;
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
                  new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
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
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }

         // we need an iterator...
         DictIterator *iter = dict->first();

         register int stackSize = vm->m_stack->size();
         vm->stackItem( stackSize - 5 ).dereference()->
               copy( iter->getCurrentKey() );
         vm->stackItem( stackSize - 4 ).dereference()->
               copy( *iter->getCurrent().dereference() );

         // prepare... iterator
         iter->setOwner( dict );
         vm->m_stack->itemAt( vm->m_stack->size()-3 ).setGCPointer( iter );
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         Sequence* seq = source->asObjectSafe()->getSequence();
         if( seq == 0 )
         {
            vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("TRAV").origin( e_orig_vm ) ) );
            goto trav_go_away;
         }

         if ( seq->empty() )
         {
            goto trav_go_away;
         }

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }

         // we need an iterator...
         CoreIterator *iter = seq->getIterator();

         *dest->dereference() = iter->getCurrent();

         // prepare... iterator
         iter->setOwner( seq );
         vm->m_stack->itemAt( vm->m_stack->size()-3 ).setGCPointer( iter );
      }
      break;

      case FLC_ITEM_STRING:
         {
         String *sstr = source->asString();

         if( sstr->length() == 0 )
            goto trav_go_away;

         // the loop may alter the string. Be sure it is in core.
         if ( ! sstr->isCore() )
         {
            sstr = new CoreString( *sstr );
            source->setString( sstr );
         }

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }
         *dest->dereference() = new CoreString( sstr->subString(0,1) );
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = ( (int64) 0 );
         }
         break;

      case FLC_ITEM_MEMBUF:
         if( source->asMemBuf()->length() == 0 )
            goto trav_go_away;

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }
         *dest->dereference() = (int64) source->asMemBuf()->get(0);
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = ( (int64) 0 );
         break;

      case FLC_ITEM_RANGE:
         if( source->asRangeIsOpen() ||
             source->asRangeEnd() == source->asRangeStart() ||
             ( source->asRangeStep() > 0 && source->asRangeStart() > source->asRangeEnd() ) ||
             ( source->asRangeStep() < 0 && source->asRangeStart() < source->asRangeEnd() )
            )
         {
            goto trav_go_away;
         }

         if( vm->operandType( 1 ) == P_PARAM_INT32 || vm->operandType( 1 ) == P_PARAM_INT64 )
         {
            vm->raiseRTError(
               new AccessError( ErrorParam( e_unpack_size ).origin( e_orig_vm ).extra( "TRAV" ) ) );
            return;
         }
         *dest->dereference() = (int64) source->asRangeStart();
         vm->m_stack->itemAt( vm->m_stack->size()-3 ) = ( (int64) source->asRangeStart() );
         break;

      case FLC_ITEM_NIL:
         // jump out
         goto trav_go_away;


      default:
         vm->raiseRTError( new TypeError( ErrorParam( e_invop ).extra("TRAV").origin( e_orig_vm ) ) );
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


// 60
void opcodeHandler_POWS( register VMachine *vm )
{
   Item *operand1 = vm->getOpcodeParam( 1 )->dereference();
   Item *operand2 = vm->getOpcodeParam( 2 );

   // TODO: S-operators
   operand1->pow( *operand2, *operand1 );
   vm->regA() = *operand1;
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
               new AccessError( ErrorParam( e_arracc ).origin( e_orig_vm ).extra( "LSB" ) ) );
      }
      return;
   }

   vm->raiseRTError(
      new TypeError( ErrorParam( e_invop ).origin( e_orig_vm ).extra( "LSB" ) ) );
}

// 0x60
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
         uint32 savePC = vm->m_pc_next;
         vm->m_pc_next = VMachine::i_pc_call_external_return;
         if( ! vm->functionalEval( *operand1 ) )
         {
            // it wasn't necessary; reset pc to the correct value
            vm->callReturn();
            vm->m_pc_next = savePC;
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

   if ( ! vm->findLocalVariable( *operand1->asString(), vm->m_regA ) )
   {
       vm->raiseRTError(
            new ParamError( ErrorParam( e_param_indir_code ).origin( e_orig_vm ).extra( "INDI" ) ) );
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

   CoreString *target = new CoreString;
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

      default:  // warning no-op
         return;
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
         if ( ! isIterator )
         {
            vm->raiseError( e_stackuf, "TRAC" );
            return;
         }
         else {
            CoreIterator *iter = (CoreIterator *) iterator->asGCPointer();

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
         //
         return;

      default:
         vm->raiseError( e_invop, "TRAC" );
         return;
   }

   /*if( copied->isString() )
      *dest = new CoreString( *copied->asString() );
   else*/
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
      vm->raiseError( e_invop, "FORB" );
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
         vm->m_regA = *operand;
         vm->m_regA.setOob( false );
         break;

      case 1:
         vm->m_regA = *operand;
         vm->m_regA.setOob( true );
         break;

      case 2:
         vm->m_regA = *operand;
         vm->m_regA.setOob( ! operand->isOob() );
         break;

      default:
         vm->m_regA.setBoolean( operand->isOob() );
   }
}

}

/* end of vm_run.cpp */
