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

#undef SRC
#define SRC "engine/psteps/stmtreturn.cpp"

#include <falcon/trace.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/textwriter.h>

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
   apply = apply_;
}


StmtReturn::StmtReturn( const StmtReturn& other ):
   Statement( other ),
   m_expr( 0 ),
   m_bHasDoubt( other.m_bHasDoubt ),
   m_bHasEval( other.m_bHasEval )
{
   FALCON_DECLARE_SYN_CLASS( stmt_return );
   apply = apply_;
   
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
   dispose( m_expr );
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
         dispose( m_expr );
         m_expr = e;
         return true;
      }
      return false;
   }
      
   dispose( m_expr );
   m_expr = 0;
   return true;
}

void StmtReturn::hasDoubt( bool b )
{
   m_bHasDoubt = b; 
}
 


void StmtReturn::hasEval( bool b )
{
   m_bHasEval = b;   
}
 

void StmtReturn::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );
   tw->write( "return" );

   if( m_bHasEval )
   {
      tw->write( "*" );
   }

   if( m_bHasDoubt )
   {
      tw->write("?" );
   }

   if( m_expr != 0 )
   {
      tw->write( " " );
      m_expr->render( tw, relativeDepth(depth) );
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}



void StmtReturn::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtReturn* self = static_cast<const StmtReturn*>( ps );
   TRACE( "StmtReturn::apply_ %s", self->describe().c_ize() );

   CodeFrame& cf = ctx->currentCode();
   int& seqId = cf.m_seqId;

   switch( seqId )
   {
   case 0:
      if( self->m_expr != 0 )
      {
         seqId = 1;
         if( ctx->stepInYield( self->m_expr, cf) )
         {
            return;
         }
         Item temp = ctx->topData();
         ctx->returnFrame(temp);
      }
      else {
         ctx->returnFrame();
      }
      break;

   case 1:
      {
         Item temp = ctx->topData();
         ctx->returnFrame(temp);
      }
      break;

   }

   // after a returnFrame, we are popped for good.
   if( self->m_bHasDoubt )
   {
      ctx->topData().setDoubt();
   }
   else {
      //TODO: Necessary?
      ctx->topData().clearDoubt();
   }

   if( self->m_bHasEval )
   {
      Class* cls = 0;
      void* data = 0;
      ctx->topData().forceClassInst(cls, data);
      cls->op_call( ctx, 0, data );
   }
}


}

/* end of stmtreturn.cpp */
