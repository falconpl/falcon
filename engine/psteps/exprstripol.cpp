/*
   FALCON - The Falcon Programming Language.
   FILE: exprstripol.cpp

   Syntactic tree item definitions -- String interpolation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 31 Jan 2013 19:30:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#include <falcon/psteps/exprstripol.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/stripoldata.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/stderrors.h>

namespace Falcon {

bool ExprStrIPol::simplify( Item& ) const
{
   return false;
}

void ExprStrIPol::apply_( const PStep* ps, VMContext* ctx )
{  
   const ExprStrIPol* self = static_cast<const ExprStrIPol*>(ps);
   TRACE( "ExprStrIPol::apply_ \"%s\"", self->describe().c_ize() );
   
   fassert( self->first() != 0 );

   CodeFrame& cf = ctx->currentCode();
   switch( cf.m_seqId )
   {
      case 0:
      self->m_mtx.lock();
      if( self->m_bTestExpr )
      {
         self->m_bTestExpr = false;
         self->m_mtx.unlock();

         // the expression is not going to change.
         if ( self->m_first->category() == TreeStep::e_cat_expression &&
                  static_cast<Expression*>(self->m_first)->trait() == Expression::e_trait_value )
         {
            const Item& value = static_cast<ExprValue*>(self->first())->item();
            if( value.isString() )
            {
               // no need to change cf.m_seqId = 1;
               // we can do everything here.
               self->handleStaticInterpolated( *value.asString(), ctx );
               return;
            }
         }
         // if we didn't return...
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_first, cf ) )
         {
           return;
         }
      }
      else if(self->m_data != 0)
      {
         self->m_mtx.unlock();
         MESSAGE1( "ExprStrIPol::apply_ Using previously created single copy" );

         ctx->pushData(Item(self->m_data->handler(), self->m_data));
         ctx->resetAndApply(&self->m_pstepIPolData);
         return;
      }
      else {
         self->m_mtx.unlock();
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_first, cf ) )
         {
           return;
         }
      }

      /* no break */

      case 1:
      {
         //cf.m_seqId = 2;
         register Item* item = &ctx->topData();
         if( ! item->isString() )
         {
            throw new OperandError(ErrorParam(e_inv_params, __LINE__)
                     .symbol("@")
                     .module("<internal>")
                     .extra("S"));
         }

         String* str = item->asString();
         self->handleDynamicInterpolated( *str, ctx );
      }
      break;
   }
}


void ExprStrIPol::handleDynamicInterpolated( const String &str, VMContext *ctx ) const
{
   TRACE1( "ExprStrIPol::handleDynamicInterpolated Creating dynamic copy of \"%s\"", describe().c_ize() );

   StrIPolData* sipol = new StrIPolData;
   int fp = 0;
   StrIPolData::t_parse_result result = sipol->parse(str, fp);

   if( result == StrIPolData::e_pr_fail )
   {
      delete sipol;
      throw new OperandError(ErrorParam(e_inv_params, __LINE__)
                           .symbol("@")
                           .module("<internal>")
                           .extra(String("Invalid interpolation at ").N(fp))
                           );
   }
   else if( result == StrIPolData::e_pr_noneed )
   {
      delete sipol; // don't need
      ctx->popCode();
      return;
   }
   else {
      ctx->topData().setUser(FALCON_GC_HANDLE(sipol));
      ctx->resetAndApply(&m_pstepIPolData);
   }
}

void ExprStrIPol::handleStaticInterpolated( const String &str, VMContext *ctx ) const
{
   TRACE1( "ExprStrIPol::handleStaticInterpolated Creating static copy of \"%s\"", describe().c_ize() );

   StrIPolData* sipol = new StrIPolData;
   int fp = 0;
   StrIPolData::t_parse_result result = sipol->parse(str, fp);

   if( result == StrIPolData::e_pr_fail )
   {
      delete sipol;
      throw new OperandError(ErrorParam(e_inv_params, __LINE__)
                           .symbol("@")
                           .module("<internal>")
                           .extra(String("Invalid interpolation at ").N(fp))
                           );
   }
   else if( result == StrIPolData::e_pr_noneed )
   {
      MESSAGE1( "ExprStrIPol::handleStaticInterpolated Detected not needed.");
      delete sipol; // don't need
      // push the value as-is
      ctx->pushData( static_cast<ExprValue*>(this->first())->item() );
      ctx->popCode();
      return;
   }
   else {
      MESSAGE1( "ExprStrIPol::handleStaticInterpolated Setting single copy.");

      m_mtx.lock();
      m_data = sipol;
      m_mtx.unlock();

      ctx->pushData(Item(sipol->handler(), sipol));
      ctx->resetAndApply(&m_pstepIPolData);
   }
}


const String& ExprStrIPol::exprName() const
{
   static String name("@");
   return name;
}


void ExprStrIPol::PStepIPolData::apply_( const PStep*, VMContext* ctx )
{
   CodeFrame& cf = ctx->currentCode();
   uint16 count = cf.m_seqId & 0xFFFF;
   uint16 depth = (cf.m_seqId>>16) & 0xFFFF;
   TRACE1( "ExprStrIPol::PStepIPolData::apply_ %d / %d", count, depth );

   Item& item = ctx->opcodeParam(depth);
   fassert( item.asClass() == StrIPolData::handler() );

   StrIPolData* sipol = static_cast<StrIPolData*>(item.asInst());
   uint32 slices = sipol->sliceCount();
   while( count < slices )
   {
      // if the slice is not static, it will leave somthing on the stack.
      if( sipol->getSlice(count)->m_type != StrIPolData::Slice::e_t_static )
      {
         ++depth;
      }

      // prepare to jump
      cf.m_seqId = (depth<<16) | (count+1);

      sipol->prepareStep(ctx, count++);
      if( &cf != &ctx->currentCode() )
      {
         TRACE1( "ExprStrIPol::PStepIPolData::apply_ returning at %d / %d", count, depth );
         return;
      }
   }

   TRACE1( "ExprStrIPol::PStepIPolData::apply_ complete at %d / %d", count, depth );

   ctx->popCode();

   String* res = sipol->mount(ctx->opcodeParams(depth));
   ctx->popData(depth+1);// we and our parent.
   ctx->pushData(FALCON_GC_HANDLE(res));
}

}

/* end of exprstripol.cpp */
