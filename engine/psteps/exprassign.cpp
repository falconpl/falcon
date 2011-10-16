/*
   FALCON - The Falcon Programming Language.
   FILE: exprassign.cpp

   Syntactic tree item definitions -- Assignment operator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/expression.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pcode.h>

namespace Falcon {


void ExprAssign::precompile( PCode* pcode ) const
{
   TRACE3( "Precompiling Assign: %p (%s)", pcode, describe().c_ize() );

   // just, evaluate the second, then evaluate the first,
   // but the first knows it's a lvalue.
   m_second->precompile(pcode);
   m_first->precompileLvalue(pcode);
}


bool ExprAssign::simplify( Item& ) const
{
   // TODO Simplify for closed symbols
   return false;
}

void ExprAssign::describeTo( String& str ) const
{
   str = "(" + m_first->describe() + " = " + m_second->describe() + ")";
}

}

/* exprassign.cpp */
