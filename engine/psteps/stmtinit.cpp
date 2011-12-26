/*
   FALCON - The Falcon Programming Language.
   FILE: stmtinit.cpp

   Statatement specialized in initialization of instances.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 12 Jul 2011 13:25:18 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/inheritance.h>
#include <falcon/vmcontext.h>
#include <falcon/falconclass.h>

#include <falcon/psteps/stmtinit.h>

namespace Falcon
{

StmtInit::StmtInit( Inheritance* inh, int32 line, int32 chr  ):
   Statement( e_stmt_init, line, chr ),
   m_postInit( this ),
   m_inheritance( inh )
{
   fassert( inh->parent() != 0 && inh->parent()->isFalconClass() );
   apply = apply_;
   inh->defineAt(line, chr);
}

StmtInit::~StmtInit()
{
   // Nothing to do
}


void StmtInit::describeTo( String& tgt, int depth ) const
{
   tgt += String(" ").replicate( depth * depthIndent ) +
         "Initialize " + m_inheritance->describe();
}


void StmtInit::oneLinerTo( String& tgt ) const
{
   tgt +=  "Initialize " + m_inheritance->describe();
}


void StmtInit::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtInit* init = static_cast<const StmtInit*>( ps );
   TRACE( "Initializing %s with %d params",
         init->m_inheritance->describe().c_ize(),
         (int) init->m_inheritance->paramCount() );
   
   // should we descend?
   ctx->resetCode( &init->m_postInit );
   if( init->m_inheritance->prepareOnContext(ctx) )
   {     
      // we went deep, let postInit to fix the thing.
      return;
   }
   
   // we shall be around just once -- this shall remove US or postInit.
   ctx->popCode();
   
   class FalconClass* fcs = static_cast<FalconClass*>( init->m_inheritance->parent());
   register CallFrame& cf = ctx->currentFrame();
   // if we're here, we didn't have any expression to evaluate.
   ctx->call( fcs->constructor(), 0, cf.m_self );
}


StmtInit::PostInit::~PostInit()
{   
}

void StmtInit::PostInit::describeTo( String& tgt, int depth ) const
{
   m_owner->describeTo(tgt, depth);
   tgt += " [postInit]";
}

void StmtInit::PostInit::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtInit* init = static_cast<const StmtInit::PostInit*>( ps )->m_owner;
   // we shall be around just once -- this shall remove US or postInit.
   ctx->popCode();
   
   class FalconClass* fcs = static_cast<FalconClass*>( init->m_inheritance->parent());
   register CallFrame& cf = ctx->currentFrame();
   ctx->call( fcs->constructor(),
              init->m_inheritance->paramCount(), cf.m_self );
}
  

}

/* end of stmtinit.cpp */
