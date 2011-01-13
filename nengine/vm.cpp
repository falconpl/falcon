/*
   FALCON - The Falcon Programming Language.
   FILE: vm.cpp

   Falcon virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 20:37:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vm.h>
#include <falcon/pcode.h>
#include <falcon/symbol.h>
#include <falcon/syntree.h>

namespace Falcon
{

VMachine::VMachine()
{
   m_codeStack.reserve( 1024 );
   m_dataStack.reserve( 1024 );
   m_callStack.reserve( 1024 );
}

VMachine::~VMachine()
{
}


bool VMachine::run()
{
   while( m_codeStack.size() >  0 )
   {
      m_codeStack.back().m_step->apply( this );
   }
}


PStep* VMachine::nextStep() const
{
}


void VMachine::call( Function* function, int nparams, const Item& self )
{
   m_callStack.push_back(
         CallFrame(function, nparams, m_dataStack.size(), m_codeStack.size(), self ) );
}


bool VMachine::step()
{
   if (  m_codeStack.size() == 0 )
   {
      return false;
   }

   const PStep* ps = m_codeStack.back().m_step;
   ps->apply( this );
}


void VMachine::returnFrame()
{
}


Item* VMachine::findLocalItem( const String& name )
{
   return 0;
}

}

/* end of vm.cpp */
