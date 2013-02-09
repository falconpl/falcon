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

#undef SRC
#define SRC "engine/rulesyntree.cpp"

#include <falcon/rulesyntree.h>
#include <falcon/vm.h>
#include <falcon/codeframe.h>
#include <falcon/statement.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

namespace Falcon
{

RuleSynTree::RuleSynTree( int line, int chr ):
   SynTree(line, chr),
   m_stepNext(this)
{
   FALCON_DECLARE_SYN_CLASS( st_rulest );
   apply = rapply_;
}

RuleSynTree::RuleSynTree( const RuleSynTree& other ):
   SynTree( other ),
   m_stepNext(this)
{
   FALCON_DECLARE_SYN_CLASS( st_rulest );
   apply = rapply_;
}

RuleSynTree::~RuleSynTree()
{
}

void RuleSynTree::rapply_( const PStep* ps, VMContext* ctx )
{
   const RuleSynTree* self = static_cast<const RuleSynTree*>(ps);
   CodeFrame& cf = ctx->currentCode();
   TRACE( "RuleSynTree::apply_ at line %d step %d/%d", self->line(), cf.m_seqId , self->size() );

   // prepare the first step
   if( self->size() == 0 )
   {
      // We did it
      ctx->pushData(Item().setBoolean(true));
      ctx->popCode();
   }
   else
   {
      // we can start the dance.
      cf.m_step = &self->m_stepNext;
      cf.m_seqId = 1;
      ctx->pushCode( self->at( 0 ) );
      ctx->clearInit();
   }   
}


void RuleSynTree::PStepNext::apply_(const PStep* ps, VMContext* ctx)
{
   const RuleSynTree::PStepNext* self = static_cast<const RuleSynTree::PStepNext*>(ps);
   // get the current step.
   CodeFrame& cf = ctx->currentCode();
   TRACE( "RuleSynTree::PStepNext::apply_ at line %d step %d", self->line(), cf.m_seqId );
   
   // Have the rule failed?
   register Item& top = ctx->topData();
   if( top.type() == FLC_ITEM_BOOL && ! top.isTrue() )
   {
      TRACE1( "RuleSynTree::PStepNext::apply_ at line %d -- failure detected", self->line());
      // Maybe we're dead?
      uint32 frameID = ctx->unrollRuleNDFrame();
      if( frameID == 0xFFFFFFFF )
      {
         // yes, we are.
         ctx->pushData(Item().setBoolean(false));
         ctx->popCode();
         return;
      }
      else
      {
         // we still have hope
         ctx->restoreInit();

         cf.m_seqId = frameID+1;
         TreeStep* step = self->m_owner->at( frameID );
         ctx->pushCode( step );
      }
   }
   else if (cf.m_seqId >= (int) self->m_owner->size() )
   {
      TRACE1( "RuleSynTree::PStepNext::apply_ at line %d -- success", self->line());
      // Commit the rule hypothesis
      ctx->commitRule();  // will pop us and our grandma StmtRule as well.
      ctx->pushData(Item().setBoolean(true));
   }
   else
   {
      // remove the return value
      bool bDoubt = ctx->topData().isDoubt();
      ctx->popData();

      // is current return non-deterministic?
      if( bDoubt )
      {
         ctx->saveInit();
         ctx->startRuleNDFrame(cf.m_seqId-1); // previous step was -1
      }
      ctx->clearInit();

      // just proceed with next step
      TreeStep* step = self->m_owner->at( cf.m_seqId++ );
      ctx->pushCode( step );
   }
}

}

/* end of rulesyntree.cpp */
