/*
   FALCON - The Falcon Programming Language.
   FILE: pcode.cpp

   Falcon virtual machine - pre-compiled code
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 17:54:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/pcode.h>
#include <falcon/vm.h>

#include <falcon/trace.h>
namespace Falcon {

PCode::PCode()
{
   apply = apply_;
}

void PCode::toString( String& res ) const
{
   if( m_steps.empty() )
   {
      res = "(<empty>)";
   }
   else {
      res = "(" + m_steps[0]->toString() + ")";
   }
}

void PCode::apply_( const PStep* self, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();

   TRACE3( "PCode apply: %p (%s)", self, self->toString().c_ize() );

   const StepList& steps = static_cast<const PCode*>(self)->m_steps;
   CodeFrame& cf = ctx->currentCode();

   // TODO Check if all this loops are really performance wise
   register int depth = ctx->codeDepth();
   register int pos = steps.size() - cf.m_seqId;
   while ( pos > 0 )
   {
      const PStep* pstep = steps[ --pos ];
      pstep->apply(pstep,vm);

      if( ctx->codeDepth() != depth )
      {
         cf.m_seqId = steps.size() - pos;
         return;
      }
   }

   // we're done?
   if( pos == 0 )
   {
      ctx->popCode();
      // save the result in the A register
      ctx->regA() = ctx->topData();
      ctx->popData();
   }
   else
   {
      cf.m_seqId = steps.size() - pos;
   }
}

}

/* end of pcode.cpp */
