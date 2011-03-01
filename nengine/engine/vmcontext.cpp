/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.cpp

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 11:36:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vmcontext.h>
#include <falcon/trace.h>
#include <stdlib.h>

namespace Falcon {

VMContext::VMContext()
{
   m_codeStack = (CodeFrame *) malloc(INITIAL_STACK_ALLOC*sizeof(CodeFrame));
   m_topCode = m_codeStack-1;
   m_maxCode = m_codeStack + INITIAL_STACK_ALLOC;

   m_callStack = (CallFrame*)  malloc(INITIAL_STACK_ALLOC*sizeof(CallFrame));
   m_topCall = m_callStack-1;
   m_maxCall = m_callStack + INITIAL_STACK_ALLOC;

   m_dataStack = (Item*) malloc(INITIAL_STACK_ALLOC*sizeof(Item));
   m_topData = m_dataStack-1;
   m_maxData = m_dataStack + INITIAL_STACK_ALLOC;
}


VMContext::VMContext( bool )
{
}

VMContext::~VMContext()
{
   free(m_codeStack);
   free(m_callStack);
   free(m_dataStack);
}


void VMContext::moreData()
{
   long distance = m_topData - m_dataStack;
   long newSize = m_maxData - m_dataStack + INCREMENT_STACK_ALLOC;
   TRACE("Reallocating %p: %d -> %ld", m_dataStack, m_maxData - m_dataStack, newSize );

   m_dataStack = (Item*) realloc( m_dataStack, newSize * sizeof(Item) );
   m_topData = m_dataStack + distance;
   m_maxData = m_dataStack + newSize;
}


void VMContext::moreCode()
{
   long distance = m_topCode - m_codeStack; // we don't want the size of the code,

   long newSize = m_maxCode - m_codeStack + INCREMENT_STACK_ALLOC;
   TRACE("Reallocating %p: %d -> %ld", m_codeStack, m_maxCode - m_codeStack, newSize );

   m_codeStack = (CodeFrame*) realloc( m_codeStack, newSize * sizeof(CodeFrame) );
   m_topCode = m_codeStack + distance;
   m_maxCode = m_codeStack + newSize;
}


void VMContext::moreCall()
{
   long distance = m_topCall - m_callStack;
   long newSize = m_maxCall - m_callStack + INCREMENT_STACK_ALLOC;
   TRACE("Reallocating %p: %d -> %ld", m_callStack, m_maxCall - m_callStack, newSize );

   m_callStack = (CallFrame*) realloc( m_callStack, newSize * sizeof(CallFrame) );
   m_topCall = m_callStack + distance;
   m_maxCall = m_callStack + newSize;
}

}

/* end of vmcontext.cpp */
