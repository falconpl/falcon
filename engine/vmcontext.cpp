/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/vm.h>
#include "vmsema.h"
#include <falcon/vmcontext.h>
#include <falcon/traits.h>
#include <falcon/genericvector.h>
#include <falcon/sys.h>

#define VM_STACK_MEMORY_THRESHOLD 256


namespace Falcon {

//==================================
// Deletor for the frame list.

VMContext::VMContext()
{
   m_sleepingOn = 0;

   m_schedule = 0.0;
   m_priority = 0;

   m_atomicMode = false;

   m_stack = new ItemVector;
   m_stack->threshHold( VM_STACK_MEMORY_THRESHOLD );
   m_stackBase = 0;

   m_tryFrame = VMachine::i_noTryFrame;

   m_pc = 0;
   m_pc_next = 0;
   m_symbol = 0;
   m_lmodule = 0;
}

VMContext::VMContext( const VMContext& other )
{
   m_sleepingOn = 0;

   m_schedule = 0.0;
   m_priority = 0;

   m_stack = new ItemVector;
   m_stack->threshHold( VM_STACK_MEMORY_THRESHOLD );
   m_stackBase = 0;

   m_tryFrame = VMachine::i_noTryFrame;

   m_pc = other.m_pc;
   m_pc_next = other.m_pc_next;
   m_symbol = other.m_symbol;
   m_lmodule = other.m_lmodule;
}

VMContext::~VMContext()
{
   delete  m_stack;
}

void VMContext::scheduleAfter( numeric secs )
{
   m_schedule = Sys::_seconds() + secs;
}


void VMContext::waitOn( VMSemaphore* sem, numeric secs )
{
   if( secs < 0.0 )
      m_schedule = -1.0;
   else
      m_schedule =  Sys::_seconds() + secs;

   m_sleepingOn = sem;
}

void VMContext::wakeup( bool signaled )
{
   if ( m_sleepingOn != 0 )  // overkill, but...
   {
      m_sleepingOn->unsubscribe( this );
      m_sleepingOn = 0;
      m_schedule = 0.0;  // immediately runnable

      // don't change the A status if not sleeping on a semaphore.
      regA().setBoolean(signaled); // we have not been awaken, and must return false
   }
}

void VMContext::signaled()
{
   if ( m_sleepingOn != 0 )  // overkill, but...
   {
      // Don't unsubscribe; the semaphore is unsubscribing us.
      m_sleepingOn = 0;
      m_schedule = 0.0;  // immediately runnable
   }

   regA().setBoolean(true); // we have not been awaken, and must return false}
}


void VMContext::createFrame( uint32 paramCount, ext_func_frame_t frameEndFunc )
{
   // space for frame
   stack().resize( stack().size() + VM_FRAME_SPACE );
   StackFrame *frame = (StackFrame *) stack().at( stack().size() - VM_FRAME_SPACE );
   frame->header.type( FLC_ITEM_INVALID );

   frame->m_symbol = symbol();
   frame->m_module = lmodule();

   frame->m_ret_pc = pc_next();
   frame->m_call_pc = pc();
   frame->m_break = false;

   frame->m_stack_base = stackBase();
   frame->m_try_base = tryFrame();

   // parameter count.
   frame->m_param_count = paramCount;

   // iterative processing support
   frame->m_endFrameFunc = frameEndFunc;

   frame->m_self.setNil();
   frame->m_binding = regBind();

   // now we can change the stack base
   stackBase() = stack().size();
}

}

/* end of vmcontext.cpp */
