/*
   FALCON - The Falcon Programming Language.
   FILE: pseudofunc.cpp

   Pseudo function definition and standard pseudo functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 17:19:48 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/pseudofunc.h>
#include <falcon/error.h>
#include <falcon/error_messages.h>
#include <falcon/vm.h>


namespace Falcon {

PseudoFunction::PseudoFunction( const String& name ):
   Function( name )
{
}

PseudoFunction::~PseudoFunction()
{
}

//========================================================
// Pseudo functions -- implementations
//

namespace PFunc {

MinOrMax::MinOrMax( const String& name, bool bIsMax ):
   PseudoFunction(name),
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

void MinOrMax::apply( VMachine* vm, int32 pCount )
{
   if( pCount != 2 )
   {
      throw paramError( __LINE__, __FILE__ );
   }

   Class* cls;
   void* udata;
   register Item* params = vm->params();
   Item& first = params[0];
   Item& second = params[1];

   if( first.asClassInst( cls, udata ) )
   {
      // ... but at worst, we must be called back.
      vm->ifDeep( &m_compareNext );
      cls->compare(udata, second );
      if( vm->wentDeep() )
      {
         return;
      }

      int comp = vm->regA().forceInteger();
      if( m_bIsMax ) comp = -comp;
      vm->regA() = *(comp <= 0 ? &first : &second);
   }
   else
   {
      int comp = first.compare(second);
      if( m_bIsMax ) comp = -comp;
      vm->regA() = *(comp <= 0 ? &first : &second);
   }

   vm->returnFrame();
}


const PStep* MinOrMax::pstep() const
{
   return &m_invoke;
}

MinOrMax::InvokeStep::InvokeStep( bool isMax ):
   m_bIsMax(isMax),
   m_compare(isMax)
{
   apply = apply_;
}

void MinOrMax::InvokeStep::apply_( const PStep* ps, VMachine* vm )
{
   Class* cls;
   void* udata;
   MinOrMax::InvokeStep* self = (MinOrMax::InvokeStep*) ps;
   
   register VMContext* ctx = vm->currentContext();
   Item& second = ctx->topData();
   Item& first = *((&ctx->topData())-1);


   if( first.asClassInst( cls, udata ) )
   {
      // ... but at worst, we must be called back.
      vm->ifDeep( &self->m_compare );
      cls->compare(udata, second );
      if( vm->wentDeep() )
      {
         return;
      }

      int comp = vm->regA().forceInteger();
      // here we're inverting the check if it's max to avoid if ! max...
      if( self->m_bIsMax ) comp = -comp;
      // ... then reversing the meaning of < with respect to max.
      vm->regA() = *(comp <= 0 ? &first : &second);
   }
   else
   {
      int comp = first.compare(second);
      if( self->m_bIsMax ) comp = -comp;
      vm->regA() = *(comp <= 0 ? &first : &second);
   }
}

MinOrMax::InvokeStep::CompareStep::CompareStep( bool isMax ):
   m_bIsMax(isMax)
{
   apply = apply_;
}

void MinOrMax::InvokeStep::CompareStep::apply_( const PStep* ps, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();
   MinOrMax::InvokeStep* self = (MinOrMax::InvokeStep*) ps;
   
   int comp = vm->regA().forceInteger();
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


MinOrMax::CompareNextStep::CompareNextStep( bool isMax ):
   m_bIsMax(isMax)
{
   apply = apply_;
}

void MinOrMax::CompareNextStep::apply_( const PStep* ps, VMachine* vm )
{
   register VMContext* ctx = vm->currentContext();
   MinOrMax::InvokeStep* self = (MinOrMax::InvokeStep*) ps;
   
   int comp = vm->regA().forceInteger();
   if( self->m_bIsMax ) comp = -comp;
   vm->regA() = *(comp <= 0 ? ctx->param(0) : ctx->param(1) );

   vm->returnFrame();
}


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

/* end of pseudofunc.cpp */
