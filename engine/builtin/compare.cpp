/*
   FALCON - The Falcon Programming Language.
   FILE: compare.cpp

   Falcon core module -- compare function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/compare.cpp"

#include <falcon/builtin/compare.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/optoken.h>

namespace Falcon {
namespace Ext {


Compare::Compare():
   PseudoFunction("compare", &m_invoke)
{
   signature("X,X");
   addParam("item");
   addParam("item2");
}

Compare::~Compare()
{
}

void Compare::invoke( VMContext* ctx, int32 )
{
   Item* item;
   Item* item2;
   
   // this is a methodic function.
   if( ctx->isMethodic() )
   {
      item = &ctx->self();
      item2 = ctx->param(0);
   }
   else
   {
      item = ctx->param(0);
      item2 = ctx->param(1);
   }

   // Checking if both the items are here.
   if( item == 0 || item2 == 0 )
   {
      throw paramError(__LINE__, "core" );
   }

   // doing the real comparison.
   Class* cls;
   void* udata;
   if( item->asClassInst( cls, udata ) )
   {
      // prepare the stack if it's not ready.
      if( ctx->isMethodic() )
      {
         ctx->pushData( *item );
         ctx->pushData( *ctx->param(0) ); // refetch after a push
      }
      // else the stack is already ok.
      
      // I don't want to be called back.
      ctx->pushCode( &m_next );
      cls->op_compare( ctx, udata );
      if( ctx->wentDeep( &m_next ) )
      {
         // wait for the return value.
         return;
      }
      ctx->popCode();

      // pass forward the topmost data.
      ctx->returnFrame( ctx->topData() );

   }
   else
   {
      ctx->returnFrame( item->compare(*item2) );
   }
}


void Compare::NextStep::apply_( const PStep*, VMContext* ctx )
{
   // pass forward the comparison result
   ctx->returnFrame(ctx->topData());
}


void Compare::Invoke::apply_( const PStep*, VMContext* ctx )
{
   Item* first, *second;
   OpToken token( ctx, first, second );
   
   // doing the real comparison.
   Class* cls;
   void* udata;
   if( first->asClassInst( cls, udata ) )
   {
      // abandon the tokens, let op_compare to do the stuff
      token.abandon();
      cls->op_compare( ctx, udata );
   }
   else
   {
      token.exit( first->compare(*second) );
   }
   ctx->popCode();
}

}
}

/* end of compare.cpp */
