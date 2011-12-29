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

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {


bool ExprAssign::simplify( Item& ) const
{
   FALCON_DECLARE_SYN_CLASS( expr_array )
   // TODO Simplify for closed symbols
   return false;
}

void ExprAssign::describeTo( String& str, int depth ) const
{
   str = "(" + m_first->describe(depth+1) + " = " + m_second->describe(depth+1) + ")";
}


void ExprAssign::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprAssign* self = static_cast<const ExprAssign*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   
   // Resolve...
   fassert( self->m_first != 0 );
   fassert( self->m_first->lvalueStep() != 0 );
   fassert( self->m_second != 0 );
   
   // Generate the values
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      case 0: 
         // check the start.
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_second, cf ) )
         {
            return;
         }
         // fallthrough
      case 1:
         cf.m_seqId = 2;
         if( ctx->stepInYield( self->m_first->lvalueStep(), cf ) )
         {
            return;
         }      
   }
   
   ctx->popCode();
}

}

/* exprassign.cpp */
