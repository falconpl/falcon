/*
   FALCON - The Falcon Programming Language.
   FILE: derivedFrom.cpp

   Falcon core module -- derivedFrom function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/derivedfrom.cpp"

#include <falcon/builtin/derivedfrom.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/errors/codeerror.h>

namespace Falcon {
namespace Ext {

DerivedFrom::DerivedFrom():
   PseudoFunction( "derivedFrom", &m_invoke )
{
   signature("X,Class");
   addParam("item");
   addParam("cls");
}

DerivedFrom::~DerivedFrom()
{
}

void DerivedFrom::invoke( VMContext* ctx, int32 nParams )
{
   Item *elem;
   Class *cls;
   if ( ctx->isMethodic() )
   {
      Item* clsItem = ctx->params();
      if( nParams < 1 || ! clsItem->isClass() )
      {
         throw paramError();
      }
      elem = &ctx->self();
      cls = static_cast<Class*>(clsItem->asInst());
   }
   else
   {
      elem = ctx->param(0);
      Item* clsItem;
      if( nParams < 2 || ! (clsItem = ctx->param(1))->isClass() )
      {
         throw paramError();
      }

      cls = static_cast<Class*>(clsItem->asInst());
   }

   Class* cls2;
   void* inst;
   elem->forceClassInst( cls2, inst );

   Item top;
   top.setBoolean(cls2->isDerivedFrom(cls));
   ctx->returnFrame( top );
}

void DerivedFrom::Invoke::apply_( const PStep*, VMContext* ctx )
{
   Item* item = ctx->opcodeParams(2);
   Item* cls = item+1;
   bool derived = false;
   if( cls->isClass() ) {
       Class* baseCls = static_cast<Class*>(cls->asInst());

       Class* cls2 = 0;
       void* inst = 0;
       item->forceClassInst( cls2, inst );
       cls2->isDerivedFrom(baseCls);
   }
   ctx->popCode();
   ctx->popData();
   ctx->topData().setBoolean(derived);
}

}
}

/* end of derivedFrom.cpp */
