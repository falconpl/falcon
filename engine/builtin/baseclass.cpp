/*
   FALCON - The Falcon Programming Language.
   FILE: baseclass.cpp

   Falcon core module -- Returns the class of an item
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 10:54:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/baseclass.cpp"

#include <falcon/builtin/baseclass.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

BaseClass::BaseClass():
   PseudoFunction( "baseClass", &m_invoke )
{
   signature("X");
   addParam("item");
}

BaseClass::~BaseClass()
{
}

void BaseClass::invoke( VMContext* ctx, int32 nParams )
{
   static Class* metaClass = Engine::instance()->metaClass();
   
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
   ctx->returnFrame( Item(metaClass, cls) );
}


void BaseClass::Invoke::apply_( const PStep*, VMContext* ctx )
{
   static Class* metaClass = Engine::instance()->metaClass();
   register Item& top = ctx->topData();
   Class* cls; void* inst;
   top.forceClassInst( cls, inst );
   top.setUser( metaClass, cls );
   ctx->popCode();
}

}
}

/* end of baseclass.cpp */


