/*
   FALCON - The Falcon Programming Language.
   FILE: foreach.cpp

   Falcon core module -- foreach function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 03 Apr 2013 00:46:54 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/foreach.cpp"

#include <falcon/builtin/foreach.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>
#include <falcon/error.h>

#include <falcon/engine.h>
#include <falcon/stdsteps.h>

namespace Falcon {
namespace Ext {



Foreach::Foreach():
   Function( "foreach" )
{
   parseDescription("sequence:X,code:C");
}

Foreach::~Foreach()
{
}

void Foreach::invoke( VMContext* ctx, int32 )
{
   class AfterCall: public PStep
   {
   public:
      AfterCall() { apply = apply_; }
      virtual ~AfterCall() {};
      void describeTo( String& s ) const { s = "ForEach::AfterCall"; }
      static void apply_( const PStep*, VMContext* ctx )
      {
         // invoke the next operator again.
         // stack is (TOP) retval, iter, sequence, code (local0)
         ctx->popCode();

         ctx->popData(); // stack is (TOP) iter, sequence, code (local0)
         Class* cls = 0;
         void* data = 0;
         // get the sequence...
         ctx->opcodeParam(1).forceClassInst( cls, data );
         cls->op_next(ctx, data);
      }
   };
   static AfterCall after_call;


   class AfterNext: public PStep
   {
   public:
      AfterNext() { apply = apply_; }
      virtual ~AfterNext() {};
      void describeTo( String& s ) const { s = "ForEach::AfterNext"; }
      static void apply_( const PStep*, VMContext* ctx )
      {
         static PStep* retTop = &Engine::instance()->stdSteps()->m_returnFrameWithTop;

         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame();
            return;
         }

         if( ! ctx->topData().isDoubt() )
         {
            // last item.
            ctx->resetCode(retTop);
         }
         else {
            ctx->pushCode(&after_call);
         }

         // call the filter.
         // stack is (TOP) value, iter, sequence, code (local0)
         Item temp = ctx->topData();
         Item code = ctx->opcodeParam(3);
         ctx->topData() = code;  // we need (TOP) value, code, ...
         ctx->pushData(temp);

         Class* cls = 0;
         void* data = 0;
         code.forceClassInst( cls, data );
         cls->op_call(ctx, 1, data);
      }
   };
   static AfterNext after_next;

   class AfterIter: public PStep
   {
   public:
      AfterIter() { apply = apply_; }
      virtual ~AfterIter() {};
      void describeTo( String& s ) const { s = "ForEach::AfterIter"; }
      static void apply_( const PStep*, VMContext* ctx )
      {
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame();
            return;
         }

         ctx->resetCode(&after_next);
         Class* cls = 0;
         void* data = 0;
         ctx->opcodeParam(1).forceClassInst( cls, data );
         cls->op_next(ctx, data);
      }
   };

   static AfterIter after_iter;


   // initialize the function
   Item *sequence;
   Item *code;

   if ( ctx->isMethodic() )
   {
      sequence = &ctx->self();
      code = ctx->param(0);
   }
   else
   {
      sequence = ctx->param(0);
      code = ctx->param(1);
   }

   if( sequence == 0 || code == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   // we need them in reverse order
   ctx->addLocals(2);
   *ctx->local(0) = *code;
   *ctx->local(1) = *sequence;
   Class* cls = 0;
   void* data = 0;
   sequence->forceClassInst( cls, data );

   ctx->pushCode( &after_iter );
   long depth = ctx->codeDepth();

   cls->op_iter(ctx, data);
   if( depth == ctx->codeDepth() )
   {
      AfterIter::apply_(&after_iter, ctx);
   }
}


}
}

/* end of foreach.cpp */
