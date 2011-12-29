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

#include <falcon/trace.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>

#include <falcon/psteps/stmtif.h>
#include <falcon/engine.h>
#include <falcon/synclasses.h>

#include <vector>

namespace Falcon
{


struct StmtIf::Private
{
    typedef std::vector< SynTree* > ElifVector;
    ElifVector m_elifs;
};

StmtIf::StmtIf( SynTree* ifTrue, SynTree* ifFalse, int32 line, int32 chr ):
   Statement( line, chr ),
   _p( new Private )
{
   static Class* mycls = &Engine::instance()->synclasses()->m_stmt_forto;
   m_class = mycls;

   apply = apply_;

   ifFalse->selector(0); // be sure that the last if is considered an else
   addElif( ifTrue );
   addElif( ifFalse );
   ifTrue->setParent(this);
   ifFalse->setParent(this);
}


StmtIf::StmtIf( SynTree* ifTrue, int32 line, int32 chr ):
   Statement( line, chr ),
   _p( new Private )
{
   static Class* mycls = &Engine::instance()->synclasses()->m_stmt_forto;
   m_class = mycls;

   apply = apply_;

   addElif( ifTrue );
   ifTrue->setParent(this);
}

StmtIf::StmtIf( int32 line, int32 chr ):
   Statement( line, chr ),
   _p( new Private )
{
   static Class* mycls = &Engine::instance()->synclasses()->m_stmt_forto;
   m_class = mycls;

   apply = apply_;
}


StmtIf::~StmtIf()
{
   // then delete our elif branches (if any)
   for( unsigned i = 0; i < _p->m_elifs.size(); ++i )
   {
      delete _p->m_elifs[i];
   }
   delete _p;
}


int32 StmtIf::arity() const
{
   return (int) _p->m_elifs.size();
}

TreeStep* StmtIf::nth( int32 n ) const
{
   int size = (int) _p->m_elifs.size();
   
   if ( n < 0 )
   {
      n = size + 1 + n;
   }
   
   if( n < 0 || n >= size ) return 0;
   return _p->m_elifs[n];
}

bool StmtIf::nth( int32 n, TreeStep* ts )
{
   int size = (int) _p->m_elifs.size();
   
   if ( n < 0 )
   {
      n = size + n;
   }
   
   if( n < 0 || n >= size ) return false;
   if ( ts != 0 && ts->category() == TreeStep::e_cat_syntree && ts->setParent(this) ) 
   {
      // change the nth elif
      delete _p->m_elifs[n];
      // if we could parent it, then it's a syntree.
      _p->m_elifs[n] = static_cast<SynTree*>(ts);
      return true;
   }   
   return false;
}

bool StmtIf::insert( int32 n, TreeStep* element )
{
   int size = (int) _p->m_elifs.size();
   
   if ( n < 0 )
   {
      n = size + n;
   }
   
   if ( element != 0 && element->category() == TreeStep::e_cat_syntree && element->setParent(this) ) 
   {
      if( n < 0 || n >= size )
      {
         _p->m_elifs.push_back( static_cast<SynTree*>(element) );
      }
      else {      
         // if we could parent it, then it's a syntree.
         _p->m_elifs.insert( _p->m_elifs.begin() + n, static_cast<SynTree*>(element));
      }
      return true;
   }   
   return false;
}

bool StmtIf::remove( int32 n )
{
   int size = (int) _p->m_elifs.size();

   if ( n < 0 )
   {
      n = size + n;
   }
   
   if( n < 0 || n >= size )
   {
      return false;
   }
   
   delete _p->m_elifs[n];
   _p->m_elifs.erase( _p->m_elifs.begin() + n );
}


StmtIf& StmtIf::addElif( SynTree* ifTrue  )
{
   _p->m_elifs.push_back( ifTrue );
   return *this;
}

void StmtIf::oneLinerTo( String& tgt ) const
{
   if( _p->m_elifs.empty() || _p->m_elifs[0]->selector() == 0 )
   {
      tgt = "if...";
   }
   else {
      tgt = "if "+ _p->m_elifs[0]->selector()->describe();
   }
}


void StmtIf::describeTo( String& tgt, int depth ) const
{
   
   String prefix = String(" ").replicate( depth * depthIndent );   
   String prefix1 = String(" ").replicate( (depth+1) * depthIndent );
   
   if( _p->m_elifs.empty() || _p->m_elifs[0]->selector() == 0 )
   {
      tgt = "if...";
      return;
   }
   
   tgt += prefix + "if "+ _p->m_elifs[0]->selector()->describe(depth+1) + "\n" +
              prefix1 + _p->m_elifs[0]->selector()->describe(depth+1) +"\n";

   for ( size_t i = 1; i < _p->m_elifs.size(); ++i )
   {      
      SynTree* tree = _p->m_elifs[i];
      if( tree->selector() != 0 )
      {
         tgt += prefix + "elif " + tree->selector()->describe(depth+1) + "\n" +
                     prefix1 + tree->describe(depth+1);
      }
      else if( i + 1 == _p->m_elifs.size() ) {
         tgt += prefix + "else\n" + prefix1 + tree->describe(depth+1) +"\n";
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
   Private::ElifVector& elifs = self->_p->m_elifs;
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
      SynTree* elif = elifs[sid++];
      if( elif->selector() == 0 )
      {
         if( sid + 1 == len )
         {
            MESSAGE1( "--Entering else" );
            ctx->resetAndApply(elif);
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
   ctx->popCode();
}

}

/* end of stmtif.cpp */
