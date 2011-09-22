/*
   FALCON - The Falcon Programming Language.
   FILE: statement.cpp

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/statement.h>
#include <falcon/expression.h>
#include <falcon/syntree.h>
#include <falcon/vm.h>

#include <falcon/codeerror.h>
#include <falcon/trace.h>

namespace Falcon
{

Breakpoint::Breakpoint( int32 line, int32 chr ):
   Statement(e_stmt_breakpoint, line, chr )
{
   apply = apply_;
}

Breakpoint::~Breakpoint()
{
}

void Breakpoint::describeTo( String& tgt ) const
{
   tgt = "(*)";
}

void Breakpoint::apply_( const PStep*, VMContext* ctx )
{
   ctx->breakpoint();
}


//====================================================================
//

StmtWhile::StmtWhile( Expression* check, SynTree* stmts, int32 line, int32 chr ):
   Statement( e_stmt_while, line, chr ),
   m_check(check),
   m_stmts( stmts )
{
   apply = apply_;
   m_bIsLoopBase = true;

   check->precompile(&m_pcCheck);
   m_pcCheck.setNextBase();

   // push ourselves and the expression in the steps
   m_step0 = this;
   m_step1 = &m_pcCheck;
}

StmtWhile::~StmtWhile()
{
   delete m_check;
   delete m_stmts;
}

void StmtWhile::oneLinerTo( String& tgt ) const
{
   tgt = "while " + m_check->describe();
}


void StmtWhile::describeTo( String& tgt ) const
{
   for( int32 i = 1; i < chr(); i++ ) {
      tgt.append(' ');
   }
   
   tgt += "while " + m_check->describe() + "\n" +
           m_stmts->describe() +
         "end\n";
}

void StmtWhile::apply_( const PStep* s1, VMContext* ctx )
{
   const StmtWhile* self = static_cast<const StmtWhile*>(s1);
   
   // break items are always nil, and so, false.
   CodeFrame& ctxTop = ctx->currentCode();
   if ( ctx->boolTopData() )
   {
      TRACE1( "Apply 'while' at line %d -- redo ", self->line() );
      // redo.
      ctx->pushCode( &self->m_pcCheck );
      ctx->pushCode( self->m_stmts );
   }
   else {
      if( &ctxTop != &ctx->currentCode() )
      {
         TRACE1( "Apply 'while' at line %d -- going deep on boolean check ", self->line() );
         return;
      }
      
      TRACE1( "Apply 'while' at line %d -- leave ", self->line() );
      //we're done
      ctx->popCode();
   }
   
   // in both cases, the data is used.
   ctx->popData();
}


StmtContinue::StmtContinue( int32 line, int32 chr ):
   Statement( e_stmt_continue, line, chr)
{
   apply = apply_;
   m_step0 = this;
}

   
void StmtContinue::describeTo( String& tgt ) const
{
   tgt = "continue";
}


void StmtContinue::apply_( const PStep*, VMContext* ctx )
{
   ctx->unrollToNextBase();
}


StmtBreak::StmtBreak( int32 line, int32 chr ):
   Statement( e_stmt_break, line, chr)
{
   apply = apply_;
   m_step0 = this;
}

   
void StmtBreak::describeTo( String& tgt ) const
{
   tgt = "break";
}


void StmtBreak::apply_( const PStep*, VMContext* ctx )
{
   ctx->unrollToLoopBase();
   Item b;
   b.setBreak();
   ctx->pushData( b );
}

//====================================================================
//

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

//============================================================
// Return
//
StmtReturn::StmtReturn( Expression* expr, int32 line, int32 chr ):
   Statement(e_stmt_return, line, chr ),
   m_expr( expr ),
   m_bHasDoubt( false )
{
   

   m_step0 = this;

   if ( expr )
   {
      m_expr = expr;
      expr->precompile( &m_pcExpr );
      m_step1 = &m_pcExpr;
      apply = apply_expr_;
   }
   else
   {
      apply = apply_;
   }
}

StmtReturn::~StmtReturn()
{
   delete m_expr;
}

void StmtReturn::expression( Expression* expr )
{
   delete m_expr;
   m_expr = expr;
   apply = m_bHasDoubt ? apply_expr_doubt_ : apply_expr_;
}


void StmtReturn::hasDoubt( bool b )
{
   m_bHasDoubt = b; 
   if( b )
   {
      apply = m_expr == 0 ? apply_expr_doubt_ : apply_doubt_;
   }
   else
   {
      apply = m_expr == 0 ? apply_expr_ : apply_;
   }
}
 

void StmtReturn::describeTo( String& tgt ) const
{
   tgt = "return";
   
   if( m_bHasDoubt )
   {
      tgt += " ?";
   }
   
   if( m_expr != 0 )
   {
      tgt += " ";
      tgt += m_expr->describe();
   }   
}


void StmtReturn::apply_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return'" );   
   ctx->returnFrame();
}


void StmtReturn::apply_expr_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return expr'" );
   ctx->returnFrame( ctx->topData() );
}

void StmtReturn::apply_doubt_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return ?'");   
   ctx->returnFrame();
   ctx->SetNDContext();
}


void StmtReturn::apply_expr_doubt_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return expr'" );
   ctx->returnFrame( ctx->topData() );
   ctx->SetNDContext();
}

}

/* end of statement.cpp */
