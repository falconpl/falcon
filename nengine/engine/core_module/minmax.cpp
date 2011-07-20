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
   signature( "X,X" );

   setDeterm(true);
}

MinOrMax::~MinOrMax()
{}

// Direct function call.
void MinOrMax::invoke( VMContext* ctx, int32 pCount )
{
   if( pCount != 2 )
   {
      throw paramError( __LINE__, __FILE__ );
   }

   Class* cls;
   void* udata;
   register Item* params = ctx->params();

   Item* op1 = params;
   Item* op2 = params + 1;

   int comp;
   if( op1->asClassInst( cls, udata ) )
   {
      // ... ask the op_compare to to the job for us.
      Item temp = *op2;
      ctx->pushData( *op1 );  // push the data...
      ctx->pushData( temp );  // and pay attention to the stack

      // ... but at worst, we must be called back.
      ctx->ifDeep( &m_compareNext );
      cls->op_compare( ctx, udata );
      if( ctx->wentDeep() )
      {
         return;
      }

      fassert( ctx->regA().isInteger() );
      comp = ctx->regA().forceInteger();

      // refetch the stack
      params = ctx->params();
      op1 = params;
      op2 = params + 1;
   }
   else
   {
      comp = op1->compare(*op2);
   }

   if( m_bIsMax )
   {
      ctx->retval( *(comp > 0 ? op1 : op2) );
   }
   else
   {
      ctx->retval( *(comp <= 0 ? op1 : op2) );
   }

   ctx->returnFrame();
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
   MinOrMax::InvokeStep* self = (MinOrMax::InvokeStep*) ps;

   int comp = ctx->regA().forceInteger();
   if( self->m_bIsMax ) comp = -comp;
   ctx->regA() = *(comp <= 0 ? ctx->param(0) : ctx->param(1) );

   ctx->returnFrame();
}


//==========================================================
//

MinOrMax::InvokeStep::InvokeStep( bool isMax ):
   m_bIsMax(isMax),
   m_compare(isMax)
{
   apply = apply_;
}

// Apply when invoked as a pseudofunction
void MinOrMax::InvokeStep::apply_( const PStep* ps, VMContext* ctx )
{
   Class* cls;
   void* udata;
   MinOrMax::InvokeStep* self = (MinOrMax::InvokeStep*) ps;
   int comp;
   Item* op1, *op2;
   ctx->operands(op1,op2);

   if( op1->asClassInst( cls, udata ) )
   {
      // ... ask the op_compare to to the job for us.
      Item temp = *op2;
      ctx->pushData( *op1 );  // push the data...
      ctx->pushData( temp );  // and pay attention to the stack

      ctx->ifDeep( &self->m_compare );
      cls->op_compare( ctx, udata );
      if( ctx->wentDeep() )
      {
         return;
      }

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
   int comp = ctx->topData().forceInteger();
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
