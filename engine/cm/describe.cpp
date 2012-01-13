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
#define SRC "falcon/cm/describe.cpp"

#include <falcon/cm/describe.h>
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

   ctx->returnFrame((new String(elem->describe()))->garbage());
}


void Describe::Invoke::apply_( const PStep*, VMContext* ctx )
{
   register Item& top = ctx->topData();
   top = (new String(top.describe()))->garbage();
   ctx->popCode();
}

}
}

/* end of describe.cpp */
