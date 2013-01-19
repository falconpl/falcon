/*
   FALCON - The Falcon Programming Language.
   FILE: exprinvoke.h

   Syntactic tree item definitions -- Invoke expression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Jan 2013 16:23:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/exprinvoke.cpp"

#include <falcon/psteps/exprinvoke.h>
#include <falcon/psteps/exprep.h>
#include <falcon/trace.h>
#include <falcon/function.h>
#include <falcon/synclasses_id.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

bool ExprInvoke::simplify( Item& ) const
{
   return false;
}

void ExprInvoke::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* tsc = Engine::instance()->treeStepClass();

   const ExprInvoke* self = static_cast<const ExprInvoke*>(ps);
   CodeFrame& cf = ctx->currentCode();
   int32 seqId = cf.m_seqId;

   TRACE2( "ExprInvoke::apply_ \"%s\" %d/3", self->describe().c_ize(), seqId );

   switch(seqId)
   {
   case 0:
      seqId = 1;
      if( ctx->stepInYield(self->m_first)) {
         return;
      }
   /* no break */

   case 1:
      seqId = 2;
      if( ctx->stepInYield(self->m_second)) {
         return;
      }
   /* no break */
   case 2:
   {
      // let's see what we have on top.
      Class* cls;
      void* data;
      if( ctx->topData().asClassInst(cls, data) )
      {
         // if it's an epex, we have to either expand or push the parameters.
         if (cls->userFlags() == FALCON_SYNCLASS_ID_EPEX)
         {
            // check if the callee is an eta
            Item* callee = &ctx->opcodeParam(1);
            bool eta = false;
            if( callee->isFunction() && callee->asFunction()->isEta() ) {
              eta = true;
            }
            else if( callee->isMethod() && callee->asMethodFunction()->isEta() ) {
              eta = true;
            }
            else if( callee->isUser() ) {
              if( callee->asClass()->typeID() == FLC_CLASS_ID_FUNC
                    && static_cast<Function*>(callee->asInst())->isEta() ){
                 eta = true;
              }
              else if( callee->asClass()->isDerivedFrom( tsc )
                       && static_cast<TreeStep*>(callee->asInst())->varmap()->isEta()  ) {
                 eta = true;
                 }
            }

            ExprEP* ep = static_cast<ExprEP*>(data);
            int32 arity = ep->arity();

            // if it's an eta, we just have to push the parameters
            if( eta )
            {
               ctx->popData(); // we don't need the expression anymore
               for( int i = 0; i < arity; ++i ) {
                  TreeStep* expr = ep->nth(i);
                  ctx->pushData(Item(expr->handler(), expr));
               }
               seqId += arity-1;
               // we'll be exiting.
               break;
            }
            else {
               // enter the parameter resolution phase
               seqId = 3;
            }
         }
         else {
            break;
         }
      }
      else {
         break;
      }
   }
   /* no break */

   default:
      // 3 and up.
   {
      int currentParam = seqId-3;
      Item* epex_i = &ctx->opcodeParam(currentParam);
      ExprEP* epex = static_cast<ExprEP*>(epex_i->asInst());
      int arity = epex->arity();

      while( currentParam < arity ) {
         seqId++;
         TreeStep* exp = epex->nth(currentParam);
         if( ctx->stepInYield(exp, cf) ) {
            return;
         }
         currentParam = seqId-3;
      }

      // we're done -- remove the expression
      ctx->removeData(currentParam,1);
      // and remove the extra parameter count, plus the extra item we removed
      seqId-=2;
   }
   break;
   }

   ctx->popCode();

   // do the call
   Item* callee = &ctx->opcodeParam(seqId-1);
   Class* cls;
   void* inst;
   callee->forceClassInst(cls, inst);
   cls->op_call(ctx, seqId-1, inst);
}

void ExprInvoke::describeTo( String& str, int depth ) const
{
   if( m_first == 0 || m_second == 0)
   {
      str = "<Blank ExprNot>";
      return;
   }

   str += "("+m_first->describe(depth+1) + " # " + m_second->describe(depth+1)+")";
}
}

/* end of exprinvoke.h */
