/*
   FALCON - The Falcon Programming Language.
   FILE: iff.cpp

   Falcon core module -- Functional if
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 13 Jan 2012 15:12:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/iff.cpp"

#include <falcon/trace.h>
#include <falcon/falcon.h>

#include <falcon/cm/iff.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>

#include <falcon/error.h>

namespace Falcon {
namespace Ext {

Iff::Iff():
   Function( "iff" )
{
   setEta(true);
   signature("X,X,[X]");
   addParam("chech");
   addParam("iftrue");
   addParam("iffalse");
}

Iff::~Iff()
{
}

void Iff::invoke( VMContext* ctx, int32 paramCount  )
{
   TRACE1( "-- called with %d params", paramCount );
   
   // all the evaluation happens in the 
   if( paramCount < 2 ) {
      throw paramError();
   }
   
   ctx->pushCode( &m_decide );
   
   Item& check = *ctx->param(0);
   Class* cls = 0;
   void* data = 0;
   check.forceClassInst( cls, data );

   ctx->pushData(check);
   // we are a function, so the context has just started.
   cls->op_call( ctx, 0, data ); 
} 
  

void Iff::PStepChoice::apply_(const PStep*, VMContext* ctx)
{
   bool result = ctx->boolTopData();
   TRACE2( "IFF Invoked PStepChoice::apply -- evaluation was : %s", 
         result ? "true" : "false" );
   
   Item* branch = result ? ctx->param(1) : ctx->param(2);
  
   if( branch != 0 )
   {
      Class* cls = 0;
      void* data = 0;
      branch->forceClassInst( cls, data );
      
#ifndef NDEBUG
      String str; cls->describe(data, str);
      TRACE2( "Returning and executi expression: %s", str.c_ize() );
#endif
      // we're out of business
      ctx->returnFrame(*branch);
      // evaluate the expression in the owner's stack.
      cls->op_call( ctx, 0, data );
   }
   else {
      MESSAGE2("Retunring nil on false");
      // we're just of business -- and our result is nil
      ctx->returnFrame();
   }
}


}
}

/* end of iff.cpp */
