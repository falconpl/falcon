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

#define VM_STACK_MEMORY_THRESHOLD 128


namespace Falcon {

//==================================
// Deletor for the frame list.

VMContext::VMContext()
{
   m_sleepingOn = 0;

   m_schedule = 0.0;
   m_priority = 0;

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

void VMContext::wakeup()
{
   if ( m_sleepingOn != 0 )
   {
      m_sleepingOn->unsubscribe( this );
      m_regA.setBoolean(false); // we have not been awaken, and must return false
      m_sleepingOn = 0; // should be done by unsubscribe, but...
   }
}

}

/* end of vmcontext.cpp */
