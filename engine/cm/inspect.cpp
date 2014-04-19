/*
   FALCON - The Falcon Programming Language.
   FILE: inspect.cpp

   Falcon core module -- inspect function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/inspect.cpp"

#include <falcon/cm/inspect.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/inspector.h>

namespace Falcon {
namespace Ext {

Inspect::Inspect():
   Function( "inspect" )
{
   signature("X,[N],[N]");
   addParam("item");
   addParam("maxdepth");
   addParam("maxsize");
}

Inspect::~Inspect()
{
}

void Inspect::invoke( VMContext* ctx, int32 )
{
   Item* i_item = ctx->param(0);
   Item* i_maxdepth = ctx->param(1);
   Item* i_maxsize = ctx->param(2);

   if( i_item == 0
            || (i_maxdepth != 0 && !i_maxdepth->isOrdinal())
            || (i_maxsize != 0 && !i_maxsize->isOrdinal())
            )
   {
      throw paramError();
   }

   // prepare the local frame
   int64 maxdepth = i_maxdepth == 0 ? 3 : i_maxdepth->forceInteger();
   int64 maxsize = i_maxsize == 0 ? -1 : i_maxsize->forceInteger();

   Inspector insp( ctx->vm()->textOut() );
   insp.inspect_r( *i_item, 0, maxdepth, maxsize );
   ctx->returnFrame();
}

}
}

/* end of inspect.cpp */
