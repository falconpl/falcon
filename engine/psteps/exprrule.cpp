/*
   FALCON - The Falcon Programming Language.
   FILE: exprrule.cpp

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
#include <falcon/synclasses_id.h>

#include <falcon/psteps/exprrule.h>

#include <vector>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

#include "exprvector_private.h"

namespace Falcon
{

class ExprRule::Private: public TSVector_Private<RuleSynTree> 
{
public:
   Private() {}
   ~Private() {}
   
   Private( const Private& other, TreeStep* owner ):
      TSVector_Private<RuleSynTree>( other, owner )
   {}
};

ExprRule::ExprRule( int32 line, int32 chr ):
   Expression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_rule );
   
   apply = apply_;
   _p = new Private;   
}

ExprRule::ExprRule( const ExprRule& other ):
         Expression( other )
{  
   apply = apply_;
   _p = new Private( *other._p, this );   
}


ExprRule::~ExprRule()
{
   delete _p;
}


ExprRule& ExprRule::addStatement( TreeStep* stmt )
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


SynTree& ExprRule::currentTree()
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


int32 ExprRule::arity() const
{
   return _p->m_exprs.size();
}

TreeStep* ExprRule::nth( int32 n ) const
{
   if (n < 0) n = _p->m_exprs.size() + n;
   if( n >= (int32) _p->m_exprs.size() )
   {
      return 0;
   }

   return _p->m_exprs[n];
}


bool ExprRule::setNth( int32 n, TreeStep* ts )
{
   if (n < 0) n = _p->m_exprs.size() + n;
   if( n > (int32) _p->m_exprs.size() )
   {
      return false;
   }
   if( n == (int32) _p->m_exprs.size() ) {
      return append(ts);
   }

   if( ts->handler()->userFlags() == FALCON_SYNCLASS_ID_RULE_SYNTREE && ts->setParent(this))
   {
      RuleSynTree* st = static_cast<RuleSynTree*>(ts);
      dispose( _p->m_exprs[n] );
      _p->m_exprs[n] = st;
      return true;
   }

   return false;
}

bool ExprRule::insert( int32 n, TreeStep* ts )
{
   if (n < 0) n = _p->m_exprs.size() + n;
   if( n > (int32) _p->m_exprs.size() )
   {
      return false;
   }
   if( n == (int32) _p->m_exprs.size() ) {
      return append(ts);
   }

   if( ts->handler()->userFlags() == FALCON_SYNCLASS_ID_RULE_SYNTREE && ts->setParent(this))
   {
      RuleSynTree* st = static_cast<RuleSynTree*>(ts);
      _p->m_exprs.insert( _p->m_exprs.begin() +n, st );
      return true;
   }

   return false;
}

bool ExprRule::append( TreeStep* ts )
{
   if( ts->handler()->userFlags() == FALCON_SYNCLASS_ID_RULE_SYNTREE && ts->setParent(this))
   {
      RuleSynTree* st = static_cast<RuleSynTree*>(ts);
      _p->m_exprs.push_back( st );
      return true;
   }

   return false;
}

bool ExprRule::remove( int32 n )
{
   if (n < 0) n = _p->m_exprs.size() + n;
   if( n > (int32) _p->m_exprs.size() )
   {
      return false;
   }

   dispose( _p->m_exprs[n] );
   _p->m_exprs.erase( _p->m_exprs.begin() + n );
   return true;
}

const SynTree& ExprRule::currentTree() const
{
   return *_p->m_exprs.back();
}

ExprRule& ExprRule::addAlternative()
{
   RuleSynTree* st = new RuleSynTree();
   st->setParent(this);
   _p->m_exprs.push_back( st );
   return *this;
}


void ExprRule::describeTo( String& tgt, int depth ) const
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


void ExprRule::oneLinerTo( String& tgt ) const
{
   if( _p->arity() == 0 )
   {
      tgt = "<Blank StmtRule>";
      return;
   }
   
   tgt += "rule ...";
}


void ExprRule::apply_( const PStep*s1 , VMContext* ctx )
{
   const ExprRule* self = static_cast<const ExprRule*>(s1);
   CodeFrame& cf = ctx->currentCode();
   TRACE( "StmtRule::apply_ at line %d step %d/%d", self->line(), cf.m_seqId , self->_p->arity() );

   if( self->_p->arity() == 0 )
   {
      // we're an immediate success
      ctx->pushData( Item().setBoolean(true) );
      ctx->popCode();
      return;
   }
   
   // initialize the rule
   if( cf.m_seqId == 0 )
   {
      ctx->startRuleFrame();
      cf.m_seqId = 1;
      RuleSynTree* st = self->_p->nth(0);
      ctx->pushCode( st );
   }
   else if (ctx->topData().isTrue() )
   {
      // success
      // But actually, this should never be called, because our branch has committed.
      TRACE( "StmtRule::apply_ at line %d -- Branch was successful (should not be here)", self->line() );
      ctx->popCode();
      // leave the result in -- but booleanize it.
      ctx->topData().setBoolean(true);
   }
   else {
      // try another branch?
      if( cf.m_seqId >= (int) self->_p->arity() )
      {
         // we're done.
         TRACE( "StmtRule::apply_ at line %d -- failed.", self->line() );
         ctx->unrollRule(); // this pops us as well.
         ctx->addDataSlot().setBoolean(false);
      }
      else {
         TRACE1( "StmtRule::apply_ at line %d -- trying next branch.", self->line() );

         RuleSynTree* next = self->_p->nth(cf.m_seqId);
         cf.m_seqId++;
         ctx->pushCode( next );
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
   dispose( m_expr );
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
      dispose( m_expr );
      m_expr = 0;
      return true;
   }
   else
   {
      if ( expr->setParent(this) ) {
         apply = apply_cut_expr_;
         dispose( m_expr );
         m_expr = expr;
         return true;
      }
   }
   
   return false;
}

   
void StmtCut::apply_( const PStep*, VMContext* ctx )
{
   ctx->dropRuleNDFrames();
   ctx->popCode();
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

   ctx->popCode(); // use us just once.
   ctx->topData().clearDoubt();
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
   dispose( m_expr );
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

Expression* StmtDoubt::selector()  const
{
   return m_expr;
}


bool StmtDoubt::selector( Expression* expr )
{
   if( expr != 0  && expr->setParent(this) )
   {
      dispose( m_expr );
      m_expr = expr;
      return true;
   }

   return false;
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
   
   ctx->popCode();
   ctx->topData().setDoubt();
}


}

/* end of exprrule.cpp */
