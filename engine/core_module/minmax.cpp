/*
   FALCON - The Falcon Programming Language.
   FILE: minmax.cpp

   Min and Max functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 22:59:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/cm/minmax.h>
#include <falcon/error.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>

namespace Falcon {
namespace Ext {

MinOrMax::MinOrMax( const String& name, bool bIsMax ):
   PseudoFunction(name, &m_invoke),
   m_bIsMax(bIsMax),
   m_invoke(bIsMax),
   m_compareNext(bIsMax)
{
   addParam("first");
   addParam("second");
   signature( "X,X,..." );
}

MinOrMax::~MinOrMax()
{}

// Direct function call.
void MinOrMax::invoke( VMContext* ctx, int32 pCount )
{
   if( pCount < 2 )
   {
      throw paramError( __LINE__, __FILE__ );
   }

   // prepare for iterative evaluation...
   ctx->pushCode( &m_compareNext );
   ctx->pushData( *ctx->param(0) );
   ctx->pushData( 0 );
   ctx->currentCode().m_seqId = 0;

   //... and call the evaluation now.
   m_compareNext.apply( &m_compareNext, ctx );
}


// Next step when called as a function
MinOrMax::CompareNextStep::CompareNextStep( bool isMax ):
   m_bIsMax(isMax)
{
   apply = apply_;
}

// Next step when called as a function
void MinOrMax::CompareNextStep::apply_( const PStep* ps, VMContext* ctx )
{
   bool m_bIsMax = static_cast<const CompareNextStep*>(ps)->m_bIsMax;
   Class* cls;
   void* udata;

   // get the indexes of current evaluations.
   int count = ctx->currentCode().m_seqId;
   const int pCount = ctx->currentFrame().m_paramCount;

   // get rid of the topmost item -- which is the result of a previous cmp
   int64 cmp = ctx->topData().forceInteger();
   ctx->popData();
   if( m_bIsMax ) cmp = -cmp;
   if( cmp < 0 )
   {
      ctx->topData() = *ctx->param( count );
   }
   
   // ... and check the next ones.
   ++ count;
   while( count < pCount )
   {
      // something complex?
      Item* current = ctx->params() + count; // this skips the check on param count
      if( current->asClassInst( cls, udata ) )
      {
         // it's a complex thing -- invoke the op_compare operator
         ctx->pushData( ctx->topData() );
         ctx->pushData( *ctx->param( count ) );
         // prepare the jump, in case we have to
         ctx->currentCode().m_seqId = count;
         cls->op_compare( ctx, udata );
         // did we went deep? -- then let the prefix to take care of next loop
         if( ctx->wentDeep( ps ) )
         {
            // let the next operator to continue
            return;
         }

         // else perform the check here.
         int64 cmp = ctx->topData().forceInteger();
         ctx->popData();
         if( m_bIsMax ) cmp = -cmp;
         if( cmp < 0 )
         {
            ctx->topData() = *ctx->param( count );
         }
      }
      else
      {
         // something simple.
         int comp = ctx->topData().compare(*current);
         if( m_bIsMax ) comp = -comp;

         if( comp > 0 )
         {
            ctx->topData() = *current;
         }
      }
      
      ++ count;
   }
   ctx->popCode();
   
   ctx->returnFrame( ctx->topData() );
}


//==========================================================
//

MinOrMax::InvokeStep::InvokeStep( bool isMax ):
   m_bIsMax(isMax),
   m_compare(isMax)
{
   apply = apply_;
}

// Apply when invoked as a pseudofunction with 2 parameters
void MinOrMax::InvokeStep::apply_( const PStep* ps, VMContext* ctx )
{
   Class* cls;
   void* udata;
   MinOrMax::InvokeStep* self = (MinOrMax::InvokeStep*) ps;
   int64 comp;
   Item* op1, *op2;
   ctx->operands(op1,op2);

   if( op1->asClassInst( cls, udata ) )
   {
      // ... ask the op_compare to to the job for us.
      Item temp = *op2;
      ctx->pushData( *op1 );  // push the data...
      ctx->pushData( temp );  // and pay attention to the stack

      ctx->pushCode( &self->m_compare );
      cls->op_compare( ctx, udata );
      if( ctx->wentDeep( &self->m_compare ) )
      {
         return;
      }
      ctx->popCode();

      fassert( ctx->topData().isInteger() );
      comp = ctx->topData().forceInteger();
      ctx->popData();
      ctx->operands(op1,op2);  // refech the stack
   }
   else
   {
      comp = op1->compare(*op2);
   }

   
   if( self->m_bIsMax )
   {
      ctx->stackResult(2, *(comp > 0 ? op1 : op2) );
   }
   else
   {
      ctx->stackResult(2, *(comp <= 0 ? op1 : op2) );
   }
}


MinOrMax::InvokeStep::CompareStep::CompareStep( bool isMax ):
   m_bIsMax(isMax)
{
   apply = apply_;
}

// Next step invoked as a pseudofunction
void MinOrMax::InvokeStep::CompareStep::apply_( const PStep* ps, VMContext* ctx )
{
   MinOrMax::InvokeStep* self = (MinOrMax::InvokeStep*) ps;

   // get the result left by the operand.
   int64 comp = ctx->topData().forceInteger();
   ctx->popData();
   if( self->m_bIsMax ) comp = -comp;

   // is the first SMALLER than the second?
   if( comp > 0 )
   {
      // then pop the second -- the stack is not destroyed
      Item& second = ctx->topData();
      ctx->popData();
      // and assign it to the first
      ctx->topData() = second;
   }
   else {
      // else just pop the second; the first is already in place.
      ctx->popData();
   }
}

//==================================================
// Min and max function declaration

Max::Max():
   MinOrMax( "max", true )
{}

Max::~Max()
{}

Min::Min():
   MinOrMax( "min", false )
{}

Min::~Min()
{}

}
}

/* end of minmax.cpp */
