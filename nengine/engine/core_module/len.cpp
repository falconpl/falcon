/*
   FALCON - The Falcon Programming Language.
   FILE: len.cpp

   Falcon core module -- len function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/cm/len.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>

namespace Falcon {
namespace Ext {

Len::Len():
   PseudoFunction( "len", &m_invoke )
{
   setDeterm(true);
   signature("X");
   addParam("item");
}

Len::~Len()
{
}

void Len::apply( VMachine* vm, int32 nParams )
{
   register VMContext* ctx = vm->currentContext();

   Item *elem;
   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
      vm->retval( elem->len() );
   }
   else
   {
      elem = ctx->param( 0 );
      if ( elem == 0 )
      {
         throw paramError();
      }
      else
      {
         vm->retval( elem->len() );
      }
   }

   vm->returnFrame();
}

void Len::Invoke::apply_( const PStep*, VMachine* vm  )
{
   register Item& top = vm->currentContext()->topData();
   top = top.len();
}

}
}

/* end of len.cpp */
