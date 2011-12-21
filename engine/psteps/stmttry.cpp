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
#define SRC "engine/psteps/stmttry.cpp"

#include <falcon/syntree.h>
#include <falcon/vmcontext.h>

#include <falcon/psteps/stmttry.h>

namespace Falcon {

StmtTry::StmtTry( SynTree* body, int32 line, int32 chr ):
   Statement( Statement::e_stmt_try, line, chr ),
   m_body( body ),
   m_fbody( 0 ),
   m_finallyStep( this )
{
   apply = apply_;
   m_bIsCatch = true;
}

StmtTry::StmtTry( int32 line, int32 chr):
   Statement( Statement::e_stmt_try, line, chr ),
   m_body( new SynTree ),
   m_fbody( 0 ),
   m_finallyStep( this )
{
   apply = apply_;
   m_bIsCatch = true;
}


StmtTry::~StmtTry()
{
   delete m_body;
   delete m_fbody;
}


void StmtTry::fbody( SynTree* body )
{
   // reset the body?
   /*
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
   */
   
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

   
void StmtTry::apply_( const PStep* ps, VMContext* ctx )
{ 
   const StmtTry* self = static_cast<const StmtTry*>(ps);
   
   
   if( self->m_body != 0 )
   {
      // push the finally step?
      if( self->m_fbody != 0 )
      {
         // put the finally processor in our place...
         ctx->resetCode( &self->m_finallyStep );
         // add a catch position placeholder
         ctx->pushCode( &self->m_stepDone );
         // ... and declare that we'll be doing some finally
         ctx->traverseFinally();
      }
      else
      {
         // just add the catch placeholder.
         ctx->resetCode( &self->m_stepDone );
      }
      
      if( ctx->stepInYield( self->m_body ) )
         return;
   }
   else if( self->m_fbody )
   {
      // we're just doing the finally!
      ctx->popCode();
      ctx->stepIn( self->m_fbody );
   }   
}

void StmtTry::PStepDone::apply_( const PStep*, VMContext* ctx )
{ 
   // we're just a placeholder for our catch clauses,
   // if we're here, then we had no throws.
   ctx->popCode(); 
}


void StmtTry::PStepFinally::apply_( const PStep* ps, VMContext* ctx )
{
   register const StmtTry* stry = static_cast<const StmtTry::PStepFinally*>(ps)->m_owner;
   
   // declare that we begin to work with finally
   ctx->enterFinally();
   ctx->currentCode().m_step = &stry->m_cleanStep;
   ctx->stepIn( stry->m_fbody );   
}


void StmtTry::PStepCleanup::apply_( const PStep*, VMContext* ctx )
{
   // declare that we're ready to be completed
   ctx->popCode();
   ctx->finallyComplete();
}


}

/* end of stmttry.cpp */
