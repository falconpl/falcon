/*
   FALCON - The Falcon Programming Language.
   FILE: sleep.h

   Falcon core module -- VMContext execution suspension
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 22 Jan 2013 11:24:20 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/sleep.cpp"

#include <falcon/trace.h>
#include <falcon/falcon.h>

#include <falcon/cm/sleep.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>

#include <falcon/error.h>

namespace Falcon {
namespace Ext {

Sleep::Sleep():
   Function( "sleep" )
{
   signature("N");
   addParam("time");
}

Sleep::~Sleep()
{
}

void Sleep::invoke( VMContext* ctx, int32 paramCount  )
{
   TRACE1( "-- called with %d params", paramCount );
   
   // all the evaluation happens in the 
   if( paramCount < 1 || ! ctx->param(0)->isOrdinal() ) {
      throw paramError();
   }
   
   numeric to = ctx->param(0)->forceNumeric();
   ctx->sleep( (int64)(to * 1000) );
   ctx->returnFrame();
} 

}
}

/* end of sleep.cpp */
