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
}

FuncPrintBase::~FuncPrintBase() {}

void FuncPrintBase::apply( VMachine* vm, int32 nParams )
{
   TRACE1("Function print%s -- apply", m_nextStep.m_isPrintl ? "l" : "" );
   m_nextStep.printNext( vm, 0 );
}

//=====================================================================
// next step
//=====================================================================

void FuncPrintBase::NextStep::apply_( const PStep* ps, VMachine* vm )
{
   fassert( vm->regA().isString() );

   const NextStep* nstep = static_cast<const NextStep*>(ps);
   TextWriter* out = vm->textOut();
   out->write( *vm->regA().asString() );
   VMContext* ctx = vm->currentContext();
   nstep->printNext( vm, ctx->currentCode().m_seqId );
}


FuncPrintBase::NextStep::NextStep()
{
   apply = apply_;
}

void FuncPrintBase::NextStep::printNext( VMachine* vm, int count ) const
{
   VMContext* ctx = vm->currentContext();
   TextWriter* out = vm->textOut();
   int nParams = ctx->currentFrame().m_paramCount;

   while( count < nParams )
   {
      Item temp;
      Class* cls;
      void* data;

      ctx->param(count)->forceClassInst( cls, data );
      ++count;

      vm->ifDeep(this);
      cls->op_toString( vm, data, temp );
      if( vm->wentDeep() )
      {
         ctx->currentCode().m_seqId = count;
         return;
      }

      TRACE3("Function print%s -- printNext", m_isPrintl ? "l" : "" );
      out->write(*temp.asString());
   }

   if (m_isPrintl)
   {
      out->write("\n");
   }

   // we're out of the function.
   vm->returnFrame();
}

}
}

/* end of print.cpp */

