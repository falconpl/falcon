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
static void s_frameDestroyer( void *sframe )
{
   delete (StackFrame *) sframe;
}

VMContext::VMContext( VMachine *origin )
{
   m_sleepingOn = 0;

   m_schedule = 0.0;
   m_priority = 0;

   m_stack = new ItemVector;
   m_stack->threshHold( VM_STACK_MEMORY_THRESHOLD );
   m_stackBase = 0;

   m_tryFrame = VMachine::i_noTryFrame;

   m_regA = origin->m_regA;
   m_regB = origin->m_regB;
   m_regS1 = origin->m_regS1;
   m_regS2 = origin->m_regS2;
   m_regL1 = origin->m_regL1;
   m_regL2 = origin->m_regL2;
   m_regBind = origin->m_regBind;

   m_symbol = origin->m_symbol;
   m_currentModule = origin->m_currentModule;
   m_currentGlobals = origin->m_currentGlobals;
   m_code = origin->m_code;
   m_pc = origin->m_pc;
   m_pc_next = origin->m_pc_next;
}

VMContext::~VMContext()
{
   delete  m_stack;
}

void VMContext::save( const VMachine *origin )
{
   m_symbol = origin->m_symbol;
   m_currentModule = origin->m_currentModule;
   m_currentGlobals = origin->m_currentGlobals;
   m_code = origin->m_code;
   m_pc = origin->m_pc;
   m_pc_next = origin->m_pc_next;

   m_regA = origin->m_regA;
   m_regB = origin->m_regB;
   m_regS1 = origin->m_regS1;
   m_regS2 = origin->m_regS2;
   m_regL1 = origin->m_regL1;
   m_regL2 = origin->m_regL2;
   m_regBind = origin->m_regBind;

   m_stackBase = origin->m_stackBase;
   m_tryFrame = origin->m_tryFrame;
}

void VMContext::restore( VMachine *origin ) const
{
   origin->m_symbol = m_symbol;
   origin->m_currentModule = m_currentModule;
   origin->m_currentGlobals = m_currentGlobals;
   origin->m_code = m_code;
   origin->m_pc = m_pc;
   origin->m_pc_next = m_pc_next;

   origin->m_stackBase = m_stackBase;
   origin->m_tryFrame = m_tryFrame;
   origin->m_stack = m_stack;

   origin->m_regA = m_regA;
   origin->m_regB = m_regB;
   origin->m_regS1 = m_regS1;
   origin->m_regS2 = m_regS2;
   origin->m_regL1 = m_regL1;
   origin->m_regL2 = m_regL2;
   origin->m_regBind = m_regBind;
}

void VMContext::wakeup()
{
   if ( m_sleepingOn != 0 )
   {
      m_sleepingOn->unsubscribe( this );
      m_sleepingOn = 0; // should be done by unsubscribe, but...
   }
}

}

/* end of vmcontext.cpp */
