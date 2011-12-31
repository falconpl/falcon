/*
   FALCON - The Falcon Programming Language.
   FILE: exprcall.cpp

   Expression controlling item (function) call
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 21:19:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/pseudofunc.h>
#include <falcon/trace.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>

#include <falcon/psteps/exprcall.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <vector>

#include "exprvector_private.h"

namespace Falcon {

ExprCall::ExprCall( int line, int chr ):
   ExprVector( line, chr ),
   m_func(0),
   m_callExpr(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
      
   apply = apply_;
}


ExprCall::ExprCall( Expression* op1, int line, int chr ):
   ExprVector( line, chr ),
   m_func(0),
   m_callExpr(op1)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
   apply = apply_;
}


ExprCall::ExprCall( PseudoFunction* f, int line, int chr ):
   ExprVector( line, chr ),
   m_func(f),
   m_callExpr(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
   apply = apply_;
}


ExprCall::ExprCall( const ExprCall& other ):
   ExprVector( other )
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
   m_func = other.m_func;
   m_callExpr = other.m_callExpr;

   apply = apply_;
}


ExprCall::~ExprCall()
{
   delete m_callExpr;
}


bool ExprCall::simplify( Item& ) const
{
   return false;
}

void ExprCall::apply_( const PStep* v, VMContext* ctx )
{
   static Engine* eng = Engine::instance();
   const ExprCall* self = static_cast<const ExprCall*>(v);
   TRACE2( "Apply CALL %s", self->describe().c_ize() );
   int pcount = self->_p->m_exprs.size();

   fassert( self->m_func != 0 || self->m_callExpr != 0 );
   
   // prepare the call expression.
   CodeFrame& cf = ctx->currentCode();
   if( cf.m_seqId == 0 )  
   {
      // got to compile or push the call item.
      cf.m_seqId = 1;
      if( self->m_callExpr != 0 )
      {
         if( ctx->stepInYield( self->m_callExpr, cf ) )
         {
            return;
         }
      }
      else 
      {
         fassert( self->m_func != 0 );
         // can we call directly our nice function?
         // call directly our pseudofunction?
         if( self->m_func->paramCount() == pcount )
         {
            ctx->resetCode( self->m_func->pstep() );
            return;
         }
         
         // Otherwise, we must handle this as a normal function
         // -- but notice that the compiler should have blocked us.         
         ctx->pushData( self->m_func );
      }
   }
   
   // now got to generate all the paraeters, if any.
   // Notice that seqId here is nparam + 1, as 0 is for the function itself.
   
   if( pcount >= cf.m_seqId )
   {
      ExprVector_Private::ExprVector::iterator pos = self->_p->m_exprs.begin() + (cf.m_seqId-1);
      ExprVector_Private::ExprVector::iterator end = self->_p->m_exprs.end();
      while( pos < end )
      {
         cf.m_seqId++;
         if( ctx->stepInYield( *pos, cf ) )
         {
            return;
         }
         ++pos;
      }
   }   
   
   // anyhow, we're out of business.
   ctx->popCode();

   // now, top points to our function value.
   register Item& top = *(&ctx->topData()-pcount);

   switch(top.type())
   {
      case FLC_ITEM_FUNC:
         {
            Function* f = top.asFunction();
            ctx->call( f, pcount );
         }
         break;

      case FLC_ITEM_METHOD:
         {
            Item old = top;
            Function* f = top.asMethodFunction();
            old.unmethodize();
            ctx->call( f, pcount, old );
         }
         break;

      case FLC_ITEM_USER:
         {
            Class* cls = top.asClass();
            void* inst = top.asInst();
            cls->op_call( ctx, pcount, inst );
         }
         break;

      default:
         {
            Class* cls = eng->getTypeClass( top.type() );
            cls->op_call( ctx, pcount, 0 );
         }
   }
}


Expression* ExprCall::selector() const
{
   return m_callExpr;
}


bool ExprCall::selector( Expression* e )
{
   if( e->setParent(this))
   {
      delete m_callExpr;
      m_callExpr = e;
      return true;
   }
   return false;
}


void ExprCall::describeTo( String& ret, int depth ) const
{
   if( m_callExpr == 0 && m_func == 0 )
   {
      ret = "<Blank ExprCall>";
      return;
   }
   
   String params;
   // and generate all the expressions, in inverse order.
   for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
   {
      if ( params.size() )
      {
         params += ", ";
      }
      params += _p->m_exprs[i]->describe(depth+1);
   }

   if( m_callExpr != 0 )
   {
      ret = m_callExpr->describe(depth+1) + "(" + params +  ")";
   }
   else
   {
      ret = m_func->name() + "(" + params +  ")";
   }
}

}

/* end of exprcall.cpp */
