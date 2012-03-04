/*
   FALCON - The Falcon Programming Language.
   FILE: classname.cpp

   Falcon core module -- Returns the name of the class of an item
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 10:54:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/classname.cpp"

#include <falcon/builtin/classname.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

ClassName::ClassName():
   PseudoFunction( "className", &m_invoke )
{
   signature("X");
   addParam("item");
}

ClassName::~ClassName()
{
}

void ClassName::invoke( VMContext* ctx, int32 nParams )
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
   
   Class* cls; void* inst;
   elem->forceClassInst( cls, inst );
   ctx->returnFrame((new String(cls->name()))->garbage());
}


void ClassName::Invoke::apply_( const PStep*, VMContext* ctx )
{
   register Item& top = ctx->topData();
   Class* cls; void* inst;
   top.forceClassInst( cls, inst );
   top = (new String(cls->name()))->garbage();
   ctx->popCode();
}

}
}

/* end of classname.cpp */

