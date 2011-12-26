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

#include <vector>

namespace Falcon
{


struct StmtIf::Private
{
    class ElifBranch;
    typedef std::vector< ElifBranch* > ElifVector;

    ElifVector m_elifs;
};


class StmtIf::Private::ElifBranch
{
public:
   Expression* m_check;
   SynTree* m_ifTrue;
   SourceRef m_sr;

   ElifBranch( Expression *check, SynTree* ifTrue, int32 line=0, int32 chr = 0 ):
      m_check( check ),
      m_ifTrue( ifTrue ),
      m_sr( line, chr )
   {
   }

   ElifBranch( const ElifBranch& other ):
      m_check( other.m_check ),
      m_ifTrue( other.m_ifTrue ),
      m_sr(other.m_sr)
   {
   }

   ~ElifBranch()
   {
      delete m_check;
      delete m_ifTrue;
   }
};


StmtIf::StmtIf( Expression* check, SynTree* ifTrue, SynTree* ifFalse, int32 line, int32 chr ):
   Statement( e_stmt_if, line, chr ),
   _p( new Private )
{
   apply = apply_;

   m_ifFalse = ifFalse;
   addElif( check, ifTrue );
}


StmtIf::~StmtIf()
{
   // if we have an else branch, delete it
   if( m_ifFalse )
   {
      delete m_ifFalse;
   }
   // then delete our elif branches (if any)
   for( unsigned i = 0; i < _p->m_elifs.size(); ++i )
   {
      delete _p->m_elifs[i];
   }
   delete _p;
}


StmtIf& StmtIf::addElif( Expression *check, SynTree* ifTrue, int32 line, int32 chr  )
{
   _p->m_elifs.push_back( new Private::ElifBranch(check, ifTrue, line, chr ) );
   return *this;
}


StmtIf& StmtIf::setElse( SynTree* ifFalse )
{
   delete m_ifFalse; // in case it was there.
   m_ifFalse = ifFalse;
   return *this;
}


void StmtIf::oneLinerTo( String& tgt ) const
{
   tgt = "if "+ _p->m_elifs[0]->m_check->describe();
}


void StmtIf::describeTo( String& tgt, int depth ) const
{
   String prefix = String(" ").replicate( depth * depthIndent );   
   String prefix1 = String(" ").replicate( (depth+1) * depthIndent );
   
   tgt += prefix + "if "+ _p->m_elifs[0]->m_check->describe(depth+1) + "\n"
              prefix1 + _p->m_elifs[0]->m_ifTrue->describe(depth+1) +"\n";

   for ( size_t i = 1; i < _p->m_elifs.size(); ++i )
   {      
      tgt += prefix + "elif " + _p->m_elifs[i]->m_check->describe(depth+1) + "\n"
                     prefix1 + _p->m_elifs[i]->m_ifTrue->describe(depth+1);
   }

   if( m_ifFalse != 0  ) {
      tgt += prefix + "else\n" + prefix1 + m_ifFalse->describe(depth+1) +"\n";
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
      ctx->resetAndApply( elifs[sid-1]->m_ifTrue );
      return;
   }
   
   // otherwise, either the branch failed or we're at 0
   // -- this is ok also if we don't have a true route.
   while( sid < len )
   {
      Private::ElifBranch* elif = elifs[sid++];

      if( ctx->stepInYield( elif->m_check, ifFrame ) )
      {
         // went deep
         return;
      }
   
      if ( ctx->boolTopDataAndPop() )
      {
         TRACE1( "--Entering elif %d", sid );
         // we're gone -- but we may use our frame.
         ctx->resetAndApply( elif->m_ifTrue );
         return;
      }
   }
   
   // all elifs failed.
   if( self->m_ifFalse != 0 )
   {
      MESSAGE1( "--Entering else" );
      ctx->resetAndApply(self->m_ifFalse);
   }
   else
   {
      // just pop
      MESSAGE1( "--Failed" );
      ctx->popCode();
   }
}

}

/* end of stmtif.cpp */
