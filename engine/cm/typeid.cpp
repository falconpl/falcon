/*
   FALCON - The Falcon Programming Language.
   FILE: typeid.cpp

   Falcon core module -- typeid function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 23:49:50 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/typeid.cpp"

#include <falcon/cm/typeid.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

TypeId::TypeId():
   PseudoFunction( "typeId", &m_invoke )
{
   signature("X");
   addParam("item");
}

TypeId::~TypeId()
{
}

void TypeId::invoke( VMContext* ctx, int32 )
{
   Item *elem;
   if ( ctx->isMethodic() )
   {
      elem = ctx->param( 0 );
      if ( elem == 0 )
      {
         throw paramError( __LINE__, "core" );
      }      
   }
   else
   {
      elem = &ctx->self();
   }

   Class* cls;
   void* data;
   
   if( elem->asClassInst( cls, data ) )
   {
      ctx->returnFrame( cls->typeID() );
   }
   else
   {
      ctx->returnFrame( elem->type() );
   }
}

void TypeId::Invoke::apply_( const PStep*, VMContext* ctx  )
{
   register Item& top = ctx->topData();

   Class* cls;
   void* data;
   if( top.asClassInst( cls, data ) )
   {
      top = cls->typeID();
   }
   else
   {
      top = top.type();
   }
   
   ctx->popCode();
}

}
}

/* end of typeid.cpp */
