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

namespace Falcon {

VMContext::VMContext()
{
   m_codeStack = new CodeFrame[1024];
   m_topCode = m_codeStack-1;

   m_callStack = new CallFrame[1024];
   m_topCall = m_callStack-1;

   m_dataStack = new Item[1024];
   m_topData = m_dataStack-1;
}


VMContext::VMContext( bool )
{

}


VMContext::~VMContext()
{
   delete[] m_codeStack;
   delete[] m_callStack;
   delete[] m_dataStack;
}

}

/* end of vmcontext.cpp */
