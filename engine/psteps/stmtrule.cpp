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

#include <falcon/engine.h>
#include <falcon/synclasses.h>

#include "exprvector_private.h"

namespace Falcon
{

class StmtRule::Private: public TSVector_Private<RuleSynTree> 
{
public:
   Private() {}
   ~Private() {}
   
   Private( const Private& other, TreeStep* owner ):
      TSVector_Private<RuleSynTree>( other, owner )
   {}
};

StmtRule::StmtRule( int32 line, int32 chr ):
   Statement( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_rule );
   
   apply = apply_;
   _p = new Private;   
}

StmtRule::StmtRule( const StmtRule& other ):
   Statement( other )
{  
   apply = apply_;   
   _p = new Private( *other._p, this );   
}


StmtRule::~StmtRule()
{
   delete _p;
}


StmtRule& StmtRule::addStatement( Statement* stmt )
{
   if( _p->arity() == 0 )
   {
      // create a base rule syntree
      RuleSynTree* st = new RuleSynTree();
      st->setParent(this);
      _p->m_exprs.push_back(st);
   }
      
   _p->m_exprs.back()->append( stmt );
   return *this;
}


SynTree& StmtRule::currentTree()
{
   if( _p->arity() == 0 )
   {
      // create a base rule syntree
      RuleSynTree* st = new RuleSynTree();
      st->setParent(this);
      _p->m_exprs.push_back(st);
   }
      
   return *_p->m_exprs.back();
}

const SynTree& StmtRule::currentTree() const
{
   return *_p->m_exprs.back();
}

StmtRule& StmtRule::addAlternative()
{
   RuleSynTree* st = new RuleSynTree();
   st->setParent(this);
   _p->m_exprs.push_back( st );
   return *this;
}


void StmtRule::describeTo( String& tgt, int depth ) const
{
   if( _p->arity() == 0 )
   {
      tgt = "<Blank StmtRule>";
      return;
   }
   
   String prefix = String( " " ).replicate( depth * depthIndent );
      
   tgt += prefix + "rule\n";
   bool bFirst = true;
   Private::ExprVector::const_iterator iter = _p->m_exprs.begin();
   while( iter != _p->m_exprs.end() )
   {
      if( ! bFirst )
      {
         tgt += prefix + "or\n";
      }
      bFirst = false;
      (*iter)->describe( depth + 1 );
      ++iter;
   }
   tgt += prefix + "end";
}


void StmtRule::oneLinerTo( String& tgt ) const
{
   if( _p->arity() == 0 )
   {
      tgt = "<Blank StmtRule>";
      return;
   }
   
   tgt += "rule ...";
}


void StmtRule::apply_( const PStep*s1 , VMContext* ctx )
{
   const StmtRule* self = static_cast<const StmtRule*>(s1);
   CodeFrame& cf = ctx->currentCode();

   fassert( self->_p->arity() > 0 );
   
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
      if( cf.m_seqId >= (int) self->_p->arity() )
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
         RuleSynTree* rst = self->_p->m_exprs[cf.m_seqId++];
         ctx->pushCode( rst );
      }
   }
}

//================================================================
// Statement cut
//

StmtCut::StmtCut( int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr(0)
{ 
   FALCON_DECLARE_SYN_CLASS( stmt_cut );
   apply = apply_;
}


StmtCut::StmtCut( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr(expr)
{
   FALCON_DECLARE_SYN_CLASS( stmt_cut );

   if( expr == 0 )
   {
      apply = apply_;
   }
   else
   {
      expr->setParent(this);
      apply = apply_cut_expr_;
   }
}

StmtCut::StmtCut( const StmtCut& other ):
   Statement( other ),
   m_expr(0)
{ 
   if( other.m_expr == 0 )
   {
      apply = apply_;
   }
   else
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);
      apply = apply_cut_expr_;
   }
}


StmtCut::~StmtCut()
{
   delete m_expr;
}

void StmtCut::describeTo( String& tgt, int depth ) const
{
   String prefix = String(" ").replicate(depth * depthIndent);
   tgt = prefix + "!";
   if( m_expr != 0 )
   {
      tgt += " ";
      tgt += m_expr->describe(depth+1);
   }
}

void StmtCut::oneLinerTo( String& tgt ) const
{
   tgt = "!";
   if( m_expr != 0 )
   {
      tgt += " ";
      tgt += m_expr->oneLiner();
   }
}


Expression* StmtCut::selector()  const
{
   return m_expr;
}


bool StmtCut::selector( Expression* expr )
{
   if( expr == 0 )
   {
      apply = apply_;
      delete m_expr;
      m_expr = 0;
      return true;
   }
   else
   {
      if ( expr->setParent(this) ) {
         apply = apply_cut_expr_;
         delete m_expr;
         m_expr = expr;
         return true;
      }
   }
   
   return false;
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

StmtDoubt::StmtDoubt( int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr(0)
{
   FALCON_DECLARE_SYN_CLASS( stmt_doubt );
   apply = apply_;
}

StmtDoubt::StmtDoubt( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr(expr)
{
   FALCON_DECLARE_SYN_CLASS( stmt_doubt );   
   apply = apply_;
   expr->setParent(this);
}


StmtDoubt::StmtDoubt( const StmtDoubt& other ):
   Statement( other ),
   m_expr(0)
{
   apply = apply_;
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);
   }
}

StmtDoubt::~StmtDoubt()
{
   delete m_expr;
}

void StmtDoubt::describeTo( String& tgt, int depth ) const
{
   if( m_expr == 0 ) {
      tgt = "<Blank StmtDoubt>";
      return;
   }
   
   tgt += String(" ").replicate( depth * depthIndent) + "? ";
   tgt += m_expr->describe( depth + 1 );
}


void StmtDoubt::oneLinerTo( String& tgt ) const
{
   if( m_expr == 0 ) {
      tgt = "<Blank StmtDoubt>";
      return;
   }
     
   tgt += "? ";
   tgt += m_expr->oneLiner();
}

void StmtDoubt::apply_( const PStep* ps, VMContext* ctx )
{  
   const StmtDoubt* self = static_cast<const StmtDoubt*>(ps);
   CodeFrame& cf = ctx->currentCode();
   
   fassert( self->m_expr != 0 );
   
   // first time around? -- call the expression.
   if( cf.m_seqId == 0 )
   {
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
