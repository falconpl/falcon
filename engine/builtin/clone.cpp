/*
   FALCON - The Falcon Programming Language.
   FILE: clone.cpp

   Falcon core module -- clone function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/clone.cpp"

#include <falcon/builtin/clone.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/stderrors.h>

namespace Falcon {
namespace Ext {

Clone::Clone():
   PseudoFunction( "clone", &m_invoke )
{
   signature("X");
   addParam("item");
}

Clone::~Clone()
{
}

void Clone::invoke( VMContext* ctx, int32 nParams )
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

   Class* cls;
   void* inst;
   elem->forceClassInst( cls, inst );
   inst = cls->clone(inst);
   if( inst == 0 )
   {
      throw new CodeError(ErrorParam( e_uncloneable, __LINE__, SRC ));
   }

   Item top( cls, inst );
   ctx->returnFrame( top );
}

void Clone::Invoke::apply_( const PStep*, VMContext* ctx )
{
   register Item& top = ctx->topData();
   Class* cls;
   void* inst;
   top.forceClassInst( cls, inst );
   inst = cls->clone(inst);
   if( inst == 0 )
   {
      throw new CodeError(ErrorParam( e_uncloneable, __LINE__, SRC ));
   }

   top.setUser( cls, inst );
   ctx->popCode();
}

}
}

/* end of clone.cpp */
