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
#include <falcon/memory.h>

#include "falcon/stackframe.h"

namespace Falcon {

VMContext::VMContext()
{
   m_codeStack = (CodeFrame *) memAlloc(INITIAL_STACK_ALLOC);
   m_topCode = m_codeStack-1;
   m_maxCode = m_codeStack + INITIAL_STACK_ALLOC;

   m_callStack = (CallFrame*)  memAlloc(INITIAL_STACK_ALLOC);
   m_topCall = m_callStack-1;
   m_maxCall = m_callStack + INITIAL_STACK_ALLOC;

   m_dataStack = (Item*) memAlloc(INITIAL_STACK_ALLOC);
   m_topData = m_dataStack-1;
   m_maxData = m_dataStack + INITIAL_STACK_ALLOC;
}


VMContext::VMContext( bool )
{
}

VMContext::~VMContext()
{
   memFree(m_codeStack);
   memFree(m_callStack);
   memFree(m_dataStack);
}


void VMContext::moreData()
{
   long distance = dataSize();
   long newSize = m_maxData - m_dataStack + INCREMENT_STACK_ALLOC;

   m_dataStack = (Item*) memRealloc( m_dataStack, newSize );
   m_topData = m_dataStack + distance;
   m_maxData = m_dataStack + newSize;
}


void VMContext::moreCode()
{
   long distance = codeDepth();
   long newSize = m_maxCode - m_codeStack + INCREMENT_STACK_ALLOC;

   m_codeStack = (CodeFrame*) memRealloc( m_codeStack, newSize );
   m_topCode = m_codeStack + distance;
   m_maxCode = m_codeStack + newSize;
}


void VMContext::moreCall()
{

   long distance = m_topCall - m_callStack;
   long newSize = m_maxCode - m_codeStack + INCREMENT_STACK_ALLOC;

   m_callStack = (CallFrame*) memRealloc( m_callStack, newSize );
   m_topCall = m_callStack + distance;
   m_maxCall = m_callStack + newSize;
}

}

/* end of vmcontext.cpp */
