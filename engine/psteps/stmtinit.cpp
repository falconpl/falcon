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
   m_inheritance( inh )
{
   fassert( inh->parent() != 0 && inh->parent()->isFalconClass() );
   apply = apply_;
   m_step0 = this;
   m_step1 = inh->compiledExpr();
   inh->defineAt(line, chr);
}

StmtInit::~StmtInit()
{
   // Nothing to do
}

void StmtInit::describeTo( String& tgt ) const
{
   tgt += "Initialize " + m_inheritance->describe();
}

void StmtInit::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtInit* init = static_cast<const StmtInit*>( ps );
   TRACE( "Initializing %s with %d params",
         init->m_inheritance->describe().c_ize(),
         (int) init->m_inheritance->paramCount() );

   // we shall be around just once
   ctx->popCode();
   class FalconClass* fcs = static_cast<FalconClass*>( init->m_inheritance->parent());
   register CallFrame& cf = ctx->currentFrame();
   ctx->call( fcs->constructor(),
              init->m_inheritance->paramCount(), cf.m_self );
}

}

/* end of stmtinit.cpp */
