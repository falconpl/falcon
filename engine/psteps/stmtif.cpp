/*
   FALCON - The Falcon Programming Language.
   FILE: stmtif.cpp

   Statatement -- if (branching)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/psteps/stmtif.cpp"

#include <falcon/trace.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>

#include <falcon/psteps/stmtif.h>
#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <vector>

#include "exprvector_private.h"

namespace Falcon
{

class StmtIf::Private: public TSVector_Private<TreeStep>
{
public:
   
   Private() {}
   ~Private() {}
   
   Private( const Private& other, TreeStep* owner ):
      TSVector_Private<TreeStep>( other, owner )
   {}
};

StmtIf::StmtIf( TreeStep* ifTrue, TreeStep* ifFalse, int32 line, int32 chr ):
   Statement( line, chr ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS(stmt_if)

   apply = apply_;

   ifFalse->selector(0); // be sure that the last if is considered an else
   addElif( ifTrue );
   addElif( ifFalse );
   ifTrue->setParent(this);
   ifFalse->setParent(this);
}


StmtIf::StmtIf( TreeStep* ifTrue, int32 line, int32 chr ):
   Statement( line, chr ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS(stmt_if)
   apply = apply_;

   addElif( ifTrue );
   ifTrue->setParent(this);
}

StmtIf::StmtIf( int32 line, int32 chr ):
   Statement( line, chr ),
   _p( new Private )
{
   FALCON_DECLARE_SYN_CLASS(stmt_if)
   apply = apply_;
}


StmtIf::StmtIf( const StmtIf& other ):
   Statement( other )
{
   apply = apply_;   
   _p =  new Private(*other._p, this);
}


StmtIf::~StmtIf()
{   
   delete _p;
}


int32 StmtIf::arity() const
{
   return _p->arity();
}

TreeStep* StmtIf::nth( int32 n ) const
{
   return _p->nth( n );
}

bool StmtIf::setNth( int32 n, TreeStep* ts )
{
   // exprvector -> nth has already nth(arity) == append semantic.
   return _p->nth( n, ts, this );
}

bool StmtIf::insert( int32 n, TreeStep* ts )
{
   // exprvector -> nth has already insert(arity) == append semantic.
   return _p->insert( n, ts, this);
}

bool StmtIf::remove( int32 n )
{
   return _p->remove( n );
}

bool StmtIf::append( TreeStep* ts )
{
   return _p->nth( _p->arity(), ts, this );
}


void StmtIf::minimize()
{
   /*Private::ExprVector& elifs = _p->m_exprs;
   uint32 len = elifs.size();
   for( uint32 i = 0; i < len; ++i ) {
      elifs[i] = minimize_basic(elifs[i]);
   }
   */
}

StmtIf& StmtIf::addElif( TreeStep* ifTrue  )
{
   if( ifTrue->setParent(this) )
   {
      _p->m_exprs.push_back( ifTrue );
   }
   return *this;
}

void StmtIf::oneLinerTo( String& tgt ) const
{
   if( _p->m_exprs.empty() || _p->m_exprs[0]->selector() == 0 )
   {
      tgt = "<Blank StmtIF>";
   }
   else {
      tgt = "if "+ _p->m_exprs[0]->selector()->describe();
   }
}


void StmtIf::describeTo( String& tgt, int depth ) const
{
   if( _p->m_exprs.empty() )
   {
      tgt = "<Blank StmtIF>";
      return;
   }
   
   String prefix = String(" ").replicate( depth * depthIndent );   
   //String prefix1 = String(" ").replicate( (depth+1) * depthIndent );
   
   if( _p->m_exprs.empty() || _p->m_exprs[0]->selector() == 0 )
   {
      tgt = "if...";
      return;
   }
   
   tgt += prefix + "if "+ _p->m_exprs[0]->selector()->describe(depth+1) + "\n" +
              _p->m_exprs[0]->describe(depth+1) +"\n";

   for ( size_t i = 1; i < _p->m_exprs.size(); ++i )
   {      
      TreeStep* tree = _p->m_exprs[i];
      if( tree->selector() != 0 )
      {
         tgt += prefix + "elif " + tree->selector()->describe(depth+1) + "\n" +
                     tree->describe(depth+1) + "\n";
      }
      else if( i + 1 == _p->m_exprs.size() ) {
         tgt += prefix + "else\n" + tree->describe(depth+1) +"\n";
      }
      else {
         tgt += "elif...\n";
      }
   }

   tgt += prefix + "end";
}


void StmtIf::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtIf* self = static_cast<const StmtIf*>(s1);
   
   CodeFrame& ifFrame = ctx->currentCode();
   Private::ExprVector& elifs = self->_p->m_exprs;
   int& sid = ifFrame.m_seqId;
   int len = (int) elifs.size();
   
   TRACE1( "Apply 'if %d/%d' at line %d ", sid, len, self->line() );
   
   if( sid > 0 && ctx->boolTopDataAndPop() )
   {
      TRACE1( "--Entering elif after descent %d", sid );
      // we're gone -- but we may use our frame.
      ctx->resetAndApply( elifs[sid-1] );
      return;
   }
   
   // otherwise, either the branch failed or we're at 0
   // -- this is ok also if we don't have a true route.
   while( sid < len )
   {
      TreeStep* elif = elifs[sid++];
      if( elif->selector() == 0 )
      {
         if( sid == len )
         {
            MESSAGE1( "--Entering else" );
            ctx->resetAndApply(elif);
            return;
         }
         else {
            MESSAGE1( "-- Ignoring elif without selector" );
         }
         continue;
      }

      if( ctx->stepInYield( elif->selector(), ifFrame ) )
      {
         // went deep
         return;
      }
   
      if ( ctx->boolTopDataAndPop() )
      {
         TRACE1( "--Entering elif %d", sid );
         // we're gone -- but we may use our frame.
         ctx->resetAndApply( elif );
         return;
      }
   }
   
   // all elifs failed.
   MESSAGE1( "--All elif failed Failed" );
   ctx->pushData(Item());
   ctx->popCode();
}

}

/* end of stmtif.cpp */
