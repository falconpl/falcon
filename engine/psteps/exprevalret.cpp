/*
   FALCON - The Falcon Programming Language.
   FILE: exprevalret.cpp

   Syntactic tree item definitions -- Return from evaluation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 05 Jul 2012 15:11:27 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/psteps/exprevalret.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/errors/operanderror.h>

namespace Falcon {


bool ExprEvalRet::simplify( Item& ) const
{
   return false;
}

void ExprEvalRet::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprEvalRet* self = static_cast<const ExprEvalRet*>(ps);
   TRACE2( "Apply \"%s\"", self->describe().c_ize() );
   
   fassert( self->first() != 0 );
   
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      case 0:
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_first, cf ) )
      {
         return;
      }
      
      // fallthrough
      case 1:
      {
         // we're done now.
         ctx->popCode();     

         Item item = ctx->topData();
         ctx->exitLocalFrame();
         ctx->pushData(item);
      }
   }
}


void ExprEvalRet::describeTo( String& str, int depth ) const
{
   if( m_first == 0 )
   {
      str = "<Blank ExprEvalRet>";
      return;
   }
   
   str = "(^= ";
   str += m_first->describe(depth+1)+")";
}

}

/* end of exprevalret.cpp */