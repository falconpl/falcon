/*
   FALCON - The Falcon Programming Language.
   FILE: stmtrule.cpp

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Apr 2011 13:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtrule.cpp"

#include <falcon/rulesyntree.h>
#include <falcon/vm.h>
#include <falcon/trace.h>

#include <falcon/psteps/stmtrule.h>

#include <vector>

namespace Falcon
{


class StmtRule::Private {
public:
   typedef std::vector<RuleSynTree*> AltTrees;
   AltTrees m_altTrees;
   
   ~Private()
   {
      AltTrees::iterator iter = m_altTrees.begin();
      while( iter != m_altTrees.end() )
      {
         delete *iter;
         ++iter;
      }
   }
};

StmtRule::StmtRule( int32 line, int32 chr ):
   Statement( e_stmt_rule, line, chr )
{
   apply = apply_;
   _p = new Private;
   // create a base rule syntree
   _p->m_altTrees.push_back( new RuleSynTree() );
}


StmtRule::~StmtRule()
{
   delete _p;
}


StmtRule& StmtRule::addStatement( Statement* stmt )
{
   _p->m_altTrees.back()->append( stmt );
   return *this;
}

SynTree& StmtRule::currentTree()
{
   return *_p->m_altTrees.back();
}

const SynTree& StmtRule::currentTree() const
{
   return *_p->m_altTrees.back();
}
StmtRule& StmtRule::addAlternative()
{
   _p->m_altTrees.push_back( new RuleSynTree() );
   return *this;
}


void StmtRule::describeTo( String& tgt ) const
{
   tgt += "rule\n";
   bool bFirst = true;
   Private::AltTrees::const_iterator iter = _p->m_altTrees.begin();
   while( iter != _p->m_altTrees.end() )
   {
      if( ! bFirst )
      {
         tgt += "or\n";
      }
      bFirst = false;
      (*iter)->describeTo(tgt);
      ++iter;
   }
   tgt += "end";
}


void StmtRule::apply_( const PStep*s1 , VMContext* ctx )
{
   const StmtRule* self = static_cast<const StmtRule*>(s1);
   CodeFrame& cf = ctx->currentCode();

   // Always process the first alternative
   if ( cf.m_seqId > 0 && ctx->ruleEntryResult() )
   {
      TRACE1( "Apply 'rule' at line %d -- success ", self->line() );      
      //we're done
      ctx->popCode();
   }
   else
   {
      // on first alternative -- or if previous alternative failed...
      if( cf.m_seqId >= (int) self->_p->m_altTrees.size() )
      {
         // we failed, and we have no more alternatives.
         TRACE1( "Apply 'rule' at line %d -- rule failed", self->line() );

         // we're done
         ctx->popCode();         
      }
      else
      {
         // we have some more alternative to try
         TRACE1( "Apply 'rule' at line %d -- applying next branch %d",
               self->line(), cf.m_seqId );

         // clear ND status
         ctx->checkNDContext();

         // create the initial rule alternative context
         ctx->startRuleFrame();

         // push the next alternative and pricess it
         RuleSynTree* rst = self->_p->m_altTrees[cf.m_seqId++];
         ctx->pushCode( rst );
      }
   }
}

//================================================================
// Statement cut
//

StmtCut::StmtCut( Expression* expr, int32 line, int32 chr ):
   Statement( e_stmt_cut,  line, chr ),
   m_expr(expr)
{
   if( expr == 0 )
   {
      apply = apply_;
   }
   else
   {
      apply = apply_cut_expr_;
   }
}

StmtCut::~StmtCut()
{
   delete m_expr;
}

void StmtCut::describeTo( String& tgt ) const
{
   tgt += "!";
   if( m_expr != 0 )
   {
      tgt += " ";
      tgt += m_expr->describe();
   }
}

void StmtCut::apply_( const PStep*, VMContext* ctx )
{
   ctx->unrollRuleBranches(); // which also pops us
}

void StmtCut::apply_cut_expr_( const PStep* ps, VMContext* ctx )
{
   CodeFrame& cf = ctx->currentCode();
   
   // first time around? -- call the expression.
   if( cf.m_seqId == 0 )
   {
      cf.m_seqId = 1;
      const StmtCut* self = static_cast<const StmtCut*>(ps);
      if( ctx->stepInYield( self->m_expr, cf ) ) 
      {
         return;
      }
   }
   // second time around? -- we have our expression solved in top data.
   
   // clear the non-deterministic bit in the context, if set.
   ctx->checkNDContext();
   ctx->popCode(); // use us just once.
   
   // we're inside a rule, or we wouldn't be called.
   register Item* td = ctx->topData().dereference();
   
   // set to false only if the last op result was a boolean false.
   ctx->ruleEntryResult( !(td->isBoolean() && td->asBoolean() == false) );
   
   // remove the data created by the expression
   ctx->popData();
}


//================================================================
// Statement doubt
//


StmtDoubt::StmtDoubt( Expression* expr, int32 line, int32 chr ):
   Statement( e_stmt_cut,  line, chr ),
   m_expr(expr)
{
   apply = apply_;
}

StmtDoubt::~StmtDoubt()
{
   delete m_expr;
}

void StmtDoubt::describeTo( String& tgt ) const
{
   tgt += "? ";
   tgt += m_expr->describe();
}

void StmtDoubt::apply_( const PStep* ps, VMContext* ctx )
{  
   CodeFrame& cf = ctx->currentCode();
   
   // first time around? -- call the expression.
   if( cf.m_seqId == 0 )
   {
      const StmtDoubt* self = static_cast<const StmtDoubt*>(ps);
      cf.m_seqId = 1;
      if( ctx->stepInYield( self->m_expr, cf ) ) 
      {
         return;
      }
   }
   
   // Declare this context as non-deterministic
   ctx->SetNDContext();
   ctx->popCode(); // use us just once.
   
   // we're inside a rule, or we wouldn't be called.
   register Item* td = ctx->topData().dereference();
   
   // set to false only if the last op result was a boolean false.
   ctx->ruleEntryResult( !(td->isBoolean() && td->asBoolean() == false) );
   
   // remove the data created by the expression
   ctx->popData();
}


}

/* end of stmtrule.cpp */
