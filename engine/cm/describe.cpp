/*
   FALCON - The Falcon Programming Language.
   FILE: describe.cpp

   Falcon core module -- describe function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/describe.cpp"

#include <falcon/cm/describe.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>
#include <falcon/errors/paramerror.h>

namespace Falcon {
namespace Ext {

Describe::Describe():
   Function( "describe" )
{
   signature("X,[N],[N]");
   //mthSignature("[N],[N]");
   addParam("item");
   addParam("depth");
   addParam("maxLength");
}

Describe::~Describe()
{
}

void Describe::invoke( VMContext* ctx, int32 )
{
   Item* elem, *md, *ml;
   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
      md = ctx->param(0);
      ml = ctx->param(1);
      if (  ( md != 0 && ! md->isOrdinal() )
         || ( ml != 0 && ! ml->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params )
            .extra("[N],[N]") );
      }
   }
   else
   {
      elem = ctx->param( 0 );
      md = ctx->param( 1 );
      ml = ctx->param( 2 );
      if ( elem == 0 
         || ( md != 0 && ! md->isOrdinal() )
         || ( ml != 0 && ! ml->isOrdinal() )
         )
      {
         throw paramError();
      }
   }

   int maxDepth = md == 0 ? 3 : md->forceInteger();
   int maxLen = ml == 0 ? 60 : ml->forceInteger();

   String* theString = new String;
   elem->describe( *theString, maxDepth, maxLen );
   ctx->returnFrame( theString->garbage() );
}

}
}

/* end of describe.cpp */
