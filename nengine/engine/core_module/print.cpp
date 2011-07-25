/*
   FALCON - The Falcon Programming Language.
   FILE: print.cpp

   Falcon core module -- print/printl functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 11:54:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/cm/print.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/textwriter.h>

namespace Falcon {
namespace Ext {

FuncPrintBase::FuncPrintBase( const String& name, bool ispl ):
   Function(name)
{
   m_nextStep.m_isPrintl = ispl;
   signature("...");
   setDeterm(true);
}

FuncPrintBase::~FuncPrintBase() {}

void FuncPrintBase::invoke( VMContext* ctx, int32 )
{
   TRACE1("Function print%s -- apply", m_nextStep.m_isPrintl ? "l" : "" );
   // [A]: create the space for op_toString
   ctx->pushData(Item());
   m_nextStep.printNext( ctx, 0 );
}

//=====================================================================
// next step
//=====================================================================

void FuncPrintBase::NextStep::apply_( const PStep* ps, VMContext* ctx )
{
   fassert( ctx->regA().isString() );

   // this is the return of a to-string deep call.
   const NextStep* nstep = static_cast<const NextStep*>(ps);
   TextWriter* out = ctx->vm()->textOut();
   // write the result of the call.
   out->write( *ctx->regA().asString() );

   // go on.
   nstep->printNext( ctx, ctx->currentCode().m_seqId );
}


FuncPrintBase::NextStep::NextStep()
{
   apply = apply_;
}

void FuncPrintBase::NextStep::printNext( VMContext* ctx, int count ) const
{
   TextWriter* out = ctx->vm()->textOut();
   int nParams = ctx->currentFrame().m_paramCount;

   // we inherit an extra topData() space from our caller (see [A])
   // push ourselves only if not already pushed and being called again.
   ctx->condPushCode(this);
   while( count < nParams )
   {
      Item* item = ctx->param(count);
      Class* cls;
      void* data;

      item->forceClassInst( cls, data );
      // put the input data for toString
      ctx->topData() = *item;
      ++count;

      ctx->currentCode().m_seqId = count;
      cls->op_toString( ctx, data );
      if( ctx->wentDeep(this) )
      {
         return;
      }

      TRACE3("Function print%s -- printNext", m_isPrintl ? "l" : "" );
      out->write(*ctx->topData().asString());
   }
   ctx->popCode();

   if (m_isPrintl)
   {
      out->write("\n");
   }

   // we're out of the function.
   ctx->returnFrame();
}

}
}

/* end of print.cpp */

