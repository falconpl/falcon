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

#include <falcon/engine.h>
#include <falcon/synclasses.h>

namespace Falcon
{

RuleSynTree::RuleSynTree():
   m_stepNext(this)
{
   FALCON_DECLARE_SYN_CLASS( st_rulest );
   apply = apply_;
}

RuleSynTree::RuleSynTree( const RuleSynTree& other ):
   SynTree( other ),
   m_stepNext(this)
{
   apply = apply_;
}

RuleSynTree::~RuleSynTree()
{
}

void RuleSynTree::apply_( const PStep* ps, VMContext* ctx )
{
   const RuleSynTree* self = static_cast<const RuleSynTree*>(ps);

   // prepare the first step
   if( self->size() == 0 )
   {
      ctx->ruleEntryResult( true );
      return;
   }
   else
   {
      // we can start the dance.
      register CodeFrame& cf = ctx->currentCode();
      cf.m_step = &self->m_stepNext;
      cf.m_seqId = 1;
      ctx->pushCode( self->at( 0 ) );
      ctx->ruleEntryResult( true );
   }   
}


void RuleSynTree::PStepNext::apply_(const PStep* ps, VMContext* ctx)
{
   const RuleSynTree::PStepNext* self = static_cast<const RuleSynTree::PStepNext*>(ps);
   // get the current step.
   CodeFrame& cf = ctx->currentCode();
   
   // Have the rule failed?
   if( ! ctx->ruleEntryResult() )
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
      ctx->ruleEntryResult( true ); // reset the result.
   }
   else if (cf.m_seqId >= (int) self->m_owner->size() )
   {
      // We have processed the rule up to the end -- SUCCESS
      ctx->popCode();
      // Commit the rule hypotesis
      ctx->commitRule();      
      return;
   }
   else if( ctx->checkNDContext() )
   {
      // we have a non-determ context at step - 1
      ctx->addRuleNDFrame( cf.m_seqId - 1 );
   }
      
   // just proceed with next step
   TreeStep* step = self->m_owner->at( cf.m_seqId++ );
   ctx->pushCode( step );
}

}

/* end of rulesyntree.cpp */
