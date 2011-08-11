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

#include <falcon/stmtrule.h>
#include <falcon/rulesyntree.h>
#include <falcon/vm.h>
#include <falcon/trace.h>

namespace Falcon
{


class StmtRule::Private {
public:
   typedef std::vector<RuleSynTree> AltTrees;
   AltTrees m_altTrees;
};

StmtRule::StmtRule( int32 line, int32 chr ):
   Statement( e_stmt_rule, line, chr )
{
   apply = apply_;
   _p = new Private;
   // create a base rule syntree
   _p->m_altTrees.push_back( RuleSynTree() );

   // push ourselves when prepare is invoked
   m_step0 = this;
}


StmtRule::~StmtRule()
{
   delete _p;
}


StmtRule& StmtRule::addStatement( Statement* stmt )
{
   _p->m_altTrees.back().append( stmt );
   return *this;
}

SynTree& StmtRule::currentTree()
{
   return _p->m_altTrees.back();
}

const SynTree& StmtRule::currentTree() const
{
   return _p->m_altTrees.back();
}
StmtRule& StmtRule::addAlternative()
{
   _p->m_altTrees.push_back( RuleSynTree() );
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
      iter->describeTo(tgt);
      ++iter;
   }
   tgt += "end";
}


void StmtRule::apply_( const PStep*s1 , VMContext* ctx )
{
   const StmtRule* self = static_cast<const StmtRule*>(s1);
   CodeFrame& cf = ctx->currentCode();

   // Always process the first alternative
   if ( cf.m_seqId > 0 && ( ! ctx->regA().isBoolean() || ctx->regA().asBoolean() == true) )
   {
      TRACE1( "Apply 'rule' at line %d -- success ", self->line() );
      // force A to be true
      ctx->regA().setBoolean(true);
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

         // force A to be false
         ctx->regA().setBoolean(false);
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
         ctx->pushCode( &self->_p->m_altTrees[cf.m_seqId++] );
      }
   }
}

//================================================================
// Statement cut

StmtCut::StmtCut( int32 line, int32 chr ):
   Statement( e_stmt_cut,  line, chr )
{
   apply = apply_;
}

StmtCut::~StmtCut()
{

}

void StmtCut::describeTo( String& tgt ) const
{
   tgt += "!";
}

void StmtCut::apply_( const PStep*, VMContext* ctx )
{
   ctx->unrollRuleBranches();
}

}

/* end of stmtrule.cpp */
