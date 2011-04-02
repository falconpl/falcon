/*
   FALCON - The Falcon Programming Language.
   FILE: rulesyntree.cpp

   Syntactic tree item definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/rulesyntree.h>
#include <falcon/vm.h>
#include <falcon/codeframe.h>
#include <falcon/statement.h>

namespace Falcon
{

RuleSynTree::RuleSynTree()
{
   apply = apply_;
}

RuleSynTree::~RuleSynTree()
{
}

void RuleSynTree::apply_( const PStep* ps, VMachine* vm )
{
   const RuleSynTree* self = static_cast<const RuleSynTree*>(ps);
   register VMContext* ctx = vm->currentContext();

   // get the current step.
   CodeFrame& cf = ctx->currentCode();

   // are we discarding our result?
   if( vm->regA().isBoolean() && vm->regA()->asBoolean() == false )
   {
      // have a we a traceback point?
      register uint32 tbpoint = ctx->unrollRuleFrame();
      if( tbpoint == 0xFFFFFFFF )
      {
         // nope. No traceback allowed -- go away with failure   
         ctx->popCode();
         return;
      }

      // we have a traceback.
      cf.m_seqId = tbpoint;
   }
   else if (cf.m_seqId >= self->m_steps.size() )
   {
      // We have processed the rule up to the end -- SUCCESS
      
      // Commit the rule hypotesis
      ctx->commitRule();
      
      ctx->popCode();
      return;
   }

   // just proceed with next step
   Statement* step = self->m_steps[ cf.m_seqId++ ];
   step->prepare(vm);
}

}

/* end of rulesyntree.cpp */
