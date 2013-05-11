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
#include <falcon/textwriter.h>

#include <falcon/psteps/exprcall.h>
#include <falcon/psteps/exprtree.h>

#include <falcon/synclasses.h>
#include <falcon/synclasses_id.h>
#include <falcon/engine.h>

#include <vector>

#include "exprvector_private.h"

namespace Falcon {

ExprCall::ExprCall( int line, int chr ):
   ExprVector( line, chr ),
   m_callExpr(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )      
   apply = apply_;
}


ExprCall::ExprCall( Expression* callee, int line, int chr ):
   ExprVector( line, chr ),
   m_callExpr(callee)
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
   apply = apply_;
}


ExprCall::ExprCall( const ExprCall& other ):
   ExprVector( other )
{
   FALCON_DECLARE_SYN_CLASS( expr_call )
   if( other.m_callExpr != 0 ) {
      m_callExpr = other.m_callExpr->clone();
      m_callExpr->setParent(this);
   }
   else {
      m_callExpr = 0;
   }
   apply = apply_;
}


ExprCall::~ExprCall()
{
   dispose( m_callExpr );
}


bool ExprCall::simplify( Item& ) const
{
   return false;
}

void ExprCall::apply_( const PStep* v, VMContext* ctx )
{
   const ExprCall* self = static_cast<const ExprCall*>(v);
   TRACE2( "Apply CALL %s", self->describe().c_ize() );
   int pcount = self->_p->m_exprs.size();

   fassert( self->m_callExpr != 0 );
   bool bHaveEta = false;
   
   // prepare the call expression.
   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      // Generate the called item.
      case 0:
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_callExpr, cf ) )
         {
            return;
         }
         /* no break */
         
      case 1:
         // Evaluate the eta-ness of the called item
         // no need to increase seqID, we won't be here with seqId=1 anymore
         if (pcount > 0)
         {
            Class* cls = 0;
            void* vts = 0;
            register Item& top = ctx->topData();
            top.forceClassInst(cls, vts);
            
            switch(cls->typeID())
            {
               case FLC_CLASS_ID_FUNC:
               {
                  Function* f = top.asFunction();
                  bHaveEta = f->isEta();
               }
               break;
               
               case FLC_ITEM_METHOD:
               {
                  Function* f = top.asMethodFunction();
                  bHaveEta = f->isEta();
               }
               break;
               
               case FLC_CLASS_ID_CLOSURE:
               {
                  Closure* cl = static_cast<Closure*>(top.asInst());
                  bHaveEta = cl->closed()->isEta();
               }
               break;

               case FLC_CLASS_ID_TREESTEP:
               {
                  if( cls->userFlags() == FALCON_SYNCLASS_ID_TREE) {
                     VarMap* st = static_cast<ExprTree*>(vts)->varmap();
                     if( st != 0 ) {
                        bHaveEta = st->isEta();
                     }
                  }
               }
            }
         }
         break;
   }     
   
   // now got to generate all the paraeters, if any.
   // Notice that seqId here is nparam + 1, as 0 is for the function itself.
   
   if( pcount >= cf.m_seqId )
   {
      TreeStepVector_Private::ExprVector::iterator pos = self->_p->m_exprs.begin() + (cf.m_seqId-1);
      TreeStepVector_Private::ExprVector::iterator end = self->_p->m_exprs.end();
      
      if( bHaveEta )
      {
         while( pos < end )
         {
            TreeStep* expr = *pos;
            ctx->pushData( Item(expr->handler(), expr ) );
            ++pos;
         }
      }
      else 
      {      
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
   }   
   
   // we're out of business
   ctx->popCode();
   ctx->caller(self);
  
   // now, top points to our function value.
   register Item& top = *(&ctx->topData()-pcount);

   switch(top.type())
   {
      case FLC_CLASS_ID_FUNC:
         {
            // this is just a shortcut for a very common case.
            Function* f = top.asFunction();
            ctx->callInternal( f, pcount );
         }
         break;

      case FLC_CLASS_ID_CLOSURE:
         {
            // this is just a shortcut for a very common case.
            Closure* f = static_cast<Closure*>(top.asInst());
            ctx->callInternal( f, pcount );
         }
         break;

      case FLC_ITEM_METHOD:
         {
            Item old = top;
            Function* f = top.asMethodFunction();
            old.unmethodize();
            ctx->callInternal( f, pcount, old );
         }
         break;

      default:
         {
            Class* cls = 0;
            void* inst = 0;
            top.forceClassInst( cls, inst );
            cls->op_call( ctx, pcount, inst );
         }
         break;
   }
}


TreeStep* ExprCall::selector() const
{
   return m_callExpr;
}


bool ExprCall::selector( TreeStep* e )
{
   if( e->setParent(this))
   {
      dispose( m_callExpr );
      m_callExpr = e;
      return true;
   }
   return false;
}


void ExprCall::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_callExpr == 0 )
   {
      tw->write("/* Blank ExprCall */");
   }
   else
   {
      m_callExpr->render( tw, relativeDepth(depth) );
      tw->write("( ");
      // and generate all the expressions, in inverse order.
      for( unsigned int i = 0; i < _p->m_exprs.size(); ++i )
      {
         if ( i > 0 )
         {
            tw->write(", ");
         }
         _p->m_exprs[i]->render(tw, relativeDepth(depth) );
      }

      tw->write(" )");
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}

}

/* end of exprcall.cpp */
