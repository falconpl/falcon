/*
   FALCON - The Falcon Programming Language.
   FILE: stmtreturn.cpp

   Statatement -- return
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
#include <falcon/stdsteps.h>

#include <falcon/psteps/stmtreturn.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

namespace Falcon
{

StmtReturn::StmtReturn( int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr( 0 ),
   m_bHasDoubt( false ),
   m_bHasEval( false )
{
   FALCON_DECLARE_SYN_CLASS( stmt_return );   
   apply = apply_;
}

StmtReturn::StmtReturn( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr( expr ),
   m_bHasDoubt( false ),
   m_bHasEval(false)
{
   FALCON_DECLARE_SYN_CLASS( stmt_return );   
   
   if ( expr )
   {
      expr->setParent(this);
      apply = apply_expr_;
   }
   else
   {
      apply = apply_;
   }
}


StmtReturn::StmtReturn( const StmtReturn& other ):
   Statement( other ),
   m_expr( 0 ),
   m_bHasDoubt( other.m_bHasDoubt ),
   m_bHasEval( other.m_bHasEval )
{
   FALCON_DECLARE_SYN_CLASS( stmt_return );   
   
   apply = other.apply;
   
   if ( other.m_expr )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);
   }
   else
   {
      m_expr = 0;
   }
}


StmtReturn::~StmtReturn()
{
   delete m_expr;
}

Expression* StmtReturn::selector() const
{
   return m_expr;
}


bool StmtReturn::selector( Expression* e )
{
   if( e!= 0  )
   {
      if( e->setParent(this) )
      {
         delete m_expr;
         m_expr = e;
         apply = m_bHasDoubt ? apply_expr_doubt_ : apply_expr_;
         return true;
      }
      return false;
   }
      
   delete m_expr;
   m_expr = 0;
   apply = m_bHasDoubt ? apply_doubt_ : apply_;
   return true;
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
 


void StmtReturn::hasEval( bool b )
{
   m_bHasEval = b;   
}
 

void StmtReturn::describeTo( String& tgt, int depth ) const
{
   tgt = String(" ").replicate(depth * depthIndent ) + "return";
   
   if( m_bHasDoubt )
   {
      tgt += " ?";
   }
   
   if( m_expr != 0 )
   {
      tgt += " ";
      tgt += m_expr->describe( depth + 1 );
   }   
}


void StmtReturn::oneLinerTo( String& tgt ) const
{
   tgt = "return";
   
   if( m_bHasDoubt )
   {
      tgt += " ?";
   }
   
   if( m_expr != 0 )
   {
      tgt += " ";
      tgt += m_expr->oneLiner();
   }   
}


void StmtReturn::apply_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return'" );   
   ctx->returnFrame();
}


void StmtReturn::apply_expr_( const PStep* ps, VMContext* ctx )
{
   static StdSteps* steps = Engine::instance()->stdSteps();
   
   MESSAGE1( "Apply 'return expr'" );
   const StmtReturn* self = static_cast<const StmtReturn*>( ps );
   
   // change our step in a standard return with top data
   if (self->m_bHasEval)
      ctx->resetCode( &steps->m_returnFrameWithTopEval );
   else
      ctx->resetCode( &steps->m_returnFrameWithTop );
   
   CodeFrame& frame = ctx->currentCode();
   ctx->stepIn( self->m_expr );
   if( &frame != &ctx->currentCode() )
   {
      // we went deep, let's the standard return frame to deal with the topic.
      return;
   }
      
   // we can return now. No need for popping, we're popping a lot here.
   ctx->returnFrame( ctx->topData() );
   
   if( self->m_bHasEval )
   {
      Class* cls = 0;
      void * data = 0;
      ctx->topData().forceClassInst( cls, data );
      cls->op_call( ctx, 0, data );
   }
}


void StmtReturn::apply_doubt_( const PStep*, VMContext* ctx )
{
   MESSAGE1( "Apply 'return ?'");   
   ctx->returnFrame();
   ctx->SetNDContext();
}


void StmtReturn::apply_expr_doubt_( const PStep* ps, VMContext* ctx )
{
   const StdSteps* steps = Engine::instance()->stdSteps();
   
   MESSAGE1( "Apply 'return expr'" );
   
   const StmtReturn* self = static_cast<const StmtReturn*>( ps );
   
   // change our step in a standard return with top data
   if (self->m_bHasEval)
      ctx->resetCode( &steps->m_returnFrameWithTopEval );
   else
      ctx->resetCode( &steps->m_returnFrameWithTop );
   
   CodeFrame& frame = ctx->currentCode();
   ctx->stepIn( self->m_expr );
   if( &frame != &ctx->currentCode() )
   {
      // we went deep, let's the standard return frame to deal with the topic.
      return;
   }
   
   ctx->returnFrame( ctx->topData() );
   ctx->SetNDContext();
   
   if( self->m_bHasEval )
   {
      Class* cls = 0;
      void * data = 0;
      ctx->topData().forceClassInst( cls, data );
      cls->op_call( ctx, 0, data );
   }
}

}

/* end of stmtreturn.cpp */
