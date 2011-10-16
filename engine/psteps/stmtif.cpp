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
   PCode m_pcCheck;
   SynTree* m_ifTrue;
   SourceRef m_sr;

   ElifBranch( Expression *check, SynTree* ifTrue, int32 line=0, int32 chr = 0 ):
      m_check( check ),
      m_ifTrue( ifTrue ),
      m_sr( line, chr )
   {
      compile();
   }

   ElifBranch( const ElifBranch& other ):
      m_check( other.m_check ),
      m_ifTrue( other.m_ifTrue ),
      m_sr(other.m_sr)
   {
      compile();
   }

   ~ElifBranch()
   {
      delete m_check;
      delete m_ifTrue;
   }

   void compile()
   {
      m_check->precompile( &m_pcCheck );
   }
};


StmtIf::StmtIf( Expression* check, SynTree* ifTrue, SynTree* ifFalse, int32 line, int32 chr ):
   Statement( e_stmt_if, line, chr ),
   _p( new Private )
{
   apply = apply_;

   m_ifFalse = ifFalse;
   addElif( check, ifTrue );
   // push ourselves and the expression in the steps
   m_step0 = this;
   m_step1 = &( _p->m_elifs[0]->m_pcCheck );
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


void StmtIf::describeTo( String& tgt ) const
{
   tgt += "if "+ _p->m_elifs[0]->m_check->describe() + "\n"
              + _p->m_elifs[0]->m_ifTrue->describe();

   for ( size_t i = 1; i < _p->m_elifs.size(); ++i )
   {      
      tgt += "elif " + _p->m_elifs[i]->m_check->describe() + "\n"
                     + _p->m_elifs[i]->m_ifTrue->describe();
   }

   if( m_ifFalse != 0  ) {
      tgt += "else\n" + m_ifFalse->describe();
   }

   tgt += "end\n";
}


void StmtIf::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtIf* self = static_cast<const StmtIf*>(s1);

   TRACE1( "Apply 'if' at line %d ", self->line() );

   int sid = ctx->currentCode().m_seqId;
   CodeFrame& ctxTop = ctx->currentCode();
   if ( ctx->boolTopData() )
   {
      ctx->popData(); // anyhow, we have consumed the data

      TRACE1( "--Entering elif %d", sid );
      // we're gone -- but we may use our frame.
      ctx->resetCode( self->_p->m_elifs[sid]->m_ifTrue );
   }
   else
   {
      if( &ctxTop != &ctx->currentCode() )
      {
         TRACE1( "Apply 'if' at line %d -- going deep on boolean check ", self->line() );
         return;
      }
      
      ctx->popData(); // anyhow, we have consumed the data

      // try next else-if
      if( ++sid < (int) self->_p->m_elifs.size() )
      {
         TRACE2( "--Trying branch %d", sid );
         ctx->currentCode().m_seqId = sid;
         ctx->pushCode( &self->_p->m_elifs[sid]->m_pcCheck );
      }
      else
      {
         // we're out of elifs.
         if( self->m_ifFalse != 0 )
         {
            MESSAGE1( "--Entering else" );
            ctx->resetCode(self->m_ifFalse);
         }
         else
         {
            // just pop
            MESSAGE1( "--Failed" );
            ctx->popCode();
         }
      }
   }
}

}

/* end of stmtif.cpp */
