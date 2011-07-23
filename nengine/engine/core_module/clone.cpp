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

#define SRC "falcon/cm/clone.cpp"

#include <falcon/cm/clone.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/codeerror.h>

namespace Falcon {
namespace Ext {

Clone::Clone():
   PseudoFunction( "clone", &m_invoke )
{
   setDeterm(true);
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

   Item top( cls, inst, true );
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

   top.setUser( cls, inst, true );
}

}
}

/* end of clone.cpp */
