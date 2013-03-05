/*
   FALCON - The Falcon Programming Language.
   FILE: extfunc.cpp

   Definition for the external function type
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/extfunc.h>
#include <falcon/vm.h>

namespace Falcon
{

void ExtFunc::invoke( VMContext* ctx, int32 pCount )
{
   ctx->regA().setNil();
   m_func(ctx, pCount );
   ctx->returnFrame(ctx->regA());
}

}

/* end of extfunc.cpp */

