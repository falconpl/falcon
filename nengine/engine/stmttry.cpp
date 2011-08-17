/*
   FALCON - The Falcon Programming Language.
   FILE: stmttry.cpp

   Syntactic tree item definitions -- Try/catch.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/stmttry.cpp"

#include <falcon/stmttry.h>
#include <falcon/syntree.h>
#include <falcon/vmcontext.h>

namespace Falcon {

StmtTry::StmtTry( SynTree* body, int32 line, int32 chr ):
   Statement( Statement::e_stmt_try, line, chr ),
   m_body( body ),
   m_fbody( 0 ),
   m_finallyStep( this )
{
   apply = apply_;
   m_bIsCatch = true;
   m_step0 = this;
   m_step1 = body;
}

StmtTry::StmtTry( int32 line, int32 chr):
   Statement( Statement::e_stmt_try, line, chr ),
   m_body( new SynTree ),
   m_fbody( 0 ),
   m_finallyStep( this )
{
   apply = apply_;
   m_bIsCatch = true;
   m_step0 = this;
   m_step1 = m_body;
}


StmtTry::~StmtTry()
{
   delete m_body;
   delete m_fbody;
}


void StmtTry::fbody( SynTree* body )
{
   // reset the body?
   if ( body == 0 )
   {
      m_step0 = this;
      m_step1 = body;
   }
   else
   {
      // set it anew
      m_step0 = &m_finallyStep;
      m_step1 = this;
      m_step2 = m_body;
      m_step3 = &m_traverseFinallyStep;
   }
   
   delete m_fbody;
   m_fbody = body;
}

void StmtTry::describeTo( String& tgt ) const
{
   // TODO: describe branches.
   tgt = "try\n" + m_body->describe() + "\nend";   
}


void StmtTry::oneLinerTo( String& tgt ) const
{
   tgt = "try ...";
}

   
void StmtTry::apply_( const PStep*, VMContext* ctx )
{ 
   // we're just a placeholder for our catch clauses,
   // if we're here, then we had no throws.
   ctx->popCode(); 
}

void StmtTry::PStepTraverse::apply_( const PStep*, VMContext* ctx )
{
   // declare that we'll be doing some finally
   ctx->popCode();
   ctx->traverseFinally();
}

void StmtTry::PStepFinally::apply_( const PStep* ps, VMContext* ctx )
{
   register const StmtTry* stry = static_cast<const StmtTry::PStepFinally*>(ps)->m_owner;
   
   // declare that we begin to work with finally
   ctx->enterFinally();
   ctx->currentCode().m_step = &stry->m_cleanStep;
   ctx->pushCode( stry->m_fbody );   
}


void StmtTry::PStepCleanup::apply_( const PStep*, VMContext* ctx )
{
   // declare that we're ready to be completed
   ctx->popCode();
   ctx->finallyComplete();
}


}

/* end of stmttry.cpp */
