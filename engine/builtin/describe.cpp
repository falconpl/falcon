/*
   FALCON - The Falcon Programming Language.
   FILE: describe.cpp

   Falcon core module -- describe function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/describe.cpp"

#include <falcon/builtin/describe.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

Describe::Describe():
   PseudoFunction( "describe", &m_invoke )
{
   signature("X");
   addParam("item");
}

Describe::~Describe()
{
}

void Describe::invoke( VMContext* ctx, int32 nParams )
{
   Item *elem;
   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
   }
   else
   {
      if( nParams <= 0 )
      {
         throw paramError();
      }

      elem = ctx->params();
   }

   ctx->returnFrame( FALCON_GC_HANDLE(new String(elem->describe())) );
}


void Describe::Invoke::apply_( const PStep*, VMContext* ctx )
{
   register Item& top = ctx->topData();
   top = FALCON_GC_HANDLE( new String(top.describe()) );
   ctx->popCode();
}

}
}

/* end of describe.cpp */
