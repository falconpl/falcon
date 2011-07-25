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

#include <algorithm>
#include <functional>


namespace Falcon {

PCode::PCode()
{
   apply = apply_;
}

void PCode::describe( String& res ) const
{
   if( m_steps.empty() )
   {
      res = "(<empty>)";
   }
   else {
      res = "(" + m_steps[0]->describe() + ")";
   }
}


void PCode::apply_( const PStep* self, VMContext* ctx )
{
   TRACE2( "PCode apply: %p (%s)", self, self->describe().c_ize() );

   const StepList& steps = static_cast<const PCode*>(self)->m_steps;
   register CodeFrame& cf = ctx->currentCode();

   // TODO Check if all this loops are really performance wise
   int size = steps.size();

   TRACE2( "PCode apply: step %d/%d", cf.m_seqId, size );

   while ( cf.m_seqId < size )
   {
      const PStep* pstep = steps[ cf.m_seqId++ ];
      pstep->apply( pstep, ctx );

      if( &ctx->currentCode() != &cf )
      {
         return;
      }
   }

   // when we're done...
   ctx->popCode();
}


}

/* end of pcode.cpp */
