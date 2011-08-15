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
   m_cleanStep( this )
{
   apply = apply_;
   // call us BEFORE our body
   m_step0 = &m_cleanStep;
   m_step1 = this;
}

StmtTry::StmtTry( int32 line, int32 chr):
   Statement( Statement::e_stmt_try, line, chr ),
   m_body( new SynTree ),
   m_fbody( 0 ),
   m_cleanStep( this )
{
   apply = apply_;
   // call us BEFORE our body
   m_step0 = &m_cleanStep;
   m_step1 = this;
}


StmtTry::~StmtTry()
{
   delete m_body;
   delete m_fbody;
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
   const StmtTry* t = static_cast<const StmtTry*>(ps);
   // we're not needed anymore in the code stack, so use our space for our body
   ctx->currentCode().m_step = t->m_body; 
   
   // creates a try frame at this position.
   // save this position
   ctx->pushTryFrame( t );
}


void StmtTry::PStepCleanup::apply_( const PStep*ps, VMContext* ctx )
{
   const StmtTry::PStepCleanup* cleanup = static_cast<const StmtTry::PStepCleanup*>(ps);
   // remove the try frame before we cause some exception here.
   ctx->popTryFrame();
   
   // have we a finally?
   SynTree* finbody = cleanup->m_owner->fbody();
   if ( finbody != 0 )
   {
      // use our space to fulfil it.
      ctx->currentCode().m_step = finbody;
   }
   else
   {
      // pop this code
      ctx->popCode();
   }
}


}

/* end of stmttry.cpp */
