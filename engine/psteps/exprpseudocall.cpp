/*
   FALCON - The Falcon Programming Language.
   FILE: exprpseudocall.cpp

   Expression controlling pseudofunction call
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 13 Jan 2012 12:46:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>

#include <falcon/psteps/exprpseudocall.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <vector>

#include "exprvector_private.h"

namespace Falcon {

ExprPseudoCall::ExprPseudoCall( int line, int chr ):
   ExprVector( line, chr ),
   m_func(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_pseudocall )
      
   apply = apply_;
}


ExprPseudoCall::ExprPseudoCall( PseudoFunction* f, int line, int chr ):
   ExprVector( line, chr ),
   m_func(f)
{
   FALCON_DECLARE_SYN_CLASS( expr_pseudocall )
   apply = f->isEta() ? apply_eta_ : apply_;
}


ExprPseudoCall::ExprPseudoCall( const ExprPseudoCall& other ):
   ExprVector( other )
{
   m_func = other.m_func;
   apply = other.apply;
}


ExprPseudoCall::~ExprPseudoCall()
{
}


bool ExprPseudoCall::simplify( Item& ) const
{
   return false;
}

void ExprPseudoCall::pseudo(PseudoFunction* ps )
{
   m_func = ps;
   apply = ps->isEta() ? apply_eta_ : apply_;
}

void ExprPseudoCall::apply_( const PStep* v, VMContext* ctx )
{
   const ExprPseudoCall* self = static_cast<const ExprPseudoCall*>(v);
   TRACE2( "Apply CALL %s", self->describe().c_ize() );  
   
   fassert( self->m_func != 0 );
   
   uint32 pcount = self->_p->m_exprs.size();
   CodeFrame& cf = ctx->currentCode();
   bool psSemantic =  self->m_func->paramCount() == pcount;
   
   // prepare the call expression.
   if( psSemantic )
   {  
      ExprVector_Private::ExprVector::iterator pos = self->_p->m_exprs.begin() + cf.m_seqId;
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
      
      // we're out of business -- invoke our nice step
      const PStep* funcStep = self->m_func->pstep();
      ctx->resetCode( funcStep );
      funcStep->apply( funcStep, ctx );
   }
   else 
   {
      // we must perform an ordinary call.
      if( cf.m_seqId == 0 )
      {
         cf.m_seqId = 1;
         ctx->pushData( self->m_func );
      }
      
      // just push the parameters
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
      
      // we're out of business
      ctx->popCode();      
      ctx->callInternal( self->m_func, pcount );
   }
}



void ExprPseudoCall::apply_eta_( const PStep* v, VMContext* ctx )
{
   const ExprPseudoCall* self = static_cast<const ExprPseudoCall*>(v);
   TRACE2( "Apply ETA CALL %s", self->describe().c_ize() );  
   
   fassert( self->m_func != 0 );   
   uint32 pcount = self->_p->m_exprs.size();
   bool psSemantic =  self->m_func->paramCount() == pcount;
   
   // prepare the call expression.
   if( ! psSemantic )
   {
      // we must perform an ordinary call.
      ctx->pushData( self->m_func );
   }
  
   ExprVector_Private::ExprVector::iterator pos = self->_p->m_exprs.begin();
   ExprVector_Private::ExprVector::iterator end = self->_p->m_exprs.end();

   while( pos < end )
   {
      Expression* expr = *pos;
      ctx->pushData( Item(expr->handler(), expr) );
      ++pos;
   }
      
   if( psSemantic )
   {
      // we're out of business -- invoke our nice step
      const PStep* funcStep = self->m_func->pstep();
      ctx->resetCode( funcStep );
      funcStep->apply( funcStep, ctx );
   }
   else 
   {
      // we're out of business -- call the function
      ctx->popCode();      
      ctx->callInternal( self->m_func, pcount );
   }
}


void ExprPseudoCall::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );
   if( m_func == 0 )
   {
     tw->write( "/* Blank 'ExprPseudoCall' */" );
   }
   else
   {
      tw->write(m_func->name());
      tw->write( "(" );
      for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
      {
         if ( i > 0 )
         {
            tw->write(", ");
         }
         _p->m_exprs[i]->render( tw, relativeDepth(depth) );
      }
      tw->write( ")" );
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}

}

/* end of ExprPseudoCall.cpp */
