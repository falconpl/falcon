/*
   FALCON - The Falcon Programming Language.
   FILE: qreturn.cpp

   Falcon core module -- qreturn function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Apr 2014 13:36:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/qreturn.cpp"

#include <falcon/builtin/qreturn.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

QReturn::QReturn():
   PseudoFunction( "returnq", &m_invoke )
{
   signature("X,X");
   addParam("check");
   addParam("value");
}

QReturn::~QReturn()
{
}

void QReturn::invoke( VMContext* ctx, int32 nParams )
{
   if( nParams < 2 )
   {
      throw paramError();
   }

   Item copy = *ctx->param(1);
   bool question = ctx->param(0)->isTrue();
   ctx->returnFrame();
   if( question )
   {
      ctx->returnFrameDoubt(copy);
   }
   else
   {
      ctx->returnFrame(copy);
   }
}

void QReturn::Invoke::apply_( const PStep*, VMContext* ctx )
{
   Item value = ctx->opcodeParam(0);
   if( ctx->opcodeParam(1).isTrue() )
   {
      ctx->returnFrameDoubt(value);
   }
   else {
      ctx->returnFrame(value);
   }
}

}
}

/* end of qreturn.cpp */
