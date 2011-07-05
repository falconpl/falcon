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

void RuleSynTree::apply_( const PStep* ps, VMContext* ctx )
{
   const RuleSynTree* self = static_cast<const RuleSynTree*>(ps);

   // get the current step.
   CodeFrame& cf = ctx->currentCode();

   // Have the rule failed?
   if( ctx->regA().isBoolean() && ctx->regA().asBoolean() == false )
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
   else if (cf.m_seqId >= (int) self->size() )
   {
      // We have processed the rule up to the end -- SUCCESS
      
      // Commit the rule hypotesis
      ctx->commitRule();      
      ctx->popCode();
      return;
   }
   else if( ctx->checkNDContext() )
   {
      // we have a non-determ context at step - 1
      ctx->addRuleNDFrame( cf.m_seqId - 1);
   }

   // just proceed with next step
   Statement* step = self->at( cf.m_seqId++ );
   step->prepare(ctx);
}

}

/* end of rulesyntree.cpp */
