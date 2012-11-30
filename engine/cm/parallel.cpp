/*
   FALCON - The Falcon Programming Language.
   FILE: parallel.cpp

   Falcon core module -- Interface to Parallel class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Nov 2012 13:52:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/parallel.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/parallel.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/contextgroup.h>
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS CGCarrier: public UserCarrier
{
public:
   ContextGroup* m_cg;

   CGCarrier( uint32 nprops, ContextGroup* grp ):
      UserCarrier(nprops)
   {
      m_cg = grp;
   }

   CGCarrier( const CGCarrier& other ):
      UserCarrier( other.dataSize() ),
      m_cg( other.m_cg )
   {
      m_cg->incref();
   }

   virtual ~CGCarrier()
   {
      m_cg->decref();
   }

   virtual CGCarrier* clone() const { return new CGCarrier(*this); }

};


ClassParallel::ClassParallel():
   ClassUser("Parallel"),

   FALCON_INIT_METHOD( wait ),
   FALCON_INIT_METHOD( timedWait ),
   FALCON_INIT_METHOD( launch ),
   FALCON_INIT_METHOD( launchWithResults )
{
}

ClassParallel::~ClassParallel()
{}


void* ClassParallel::createInstance() const
{
   CGCarrier* ctg = new CGCarrier(carriedProps(), new ContextGroup );
   return ctg;
}


bool ClassParallel::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   if ( pcount == 0 )
   {
      throw new ParamError( ErrorParam(e_inv_params, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime )
               .extra("C,...") );
   }

   CGCarrier* ctg = static_cast<CGCarrier*>(instance);
   Item* params = ctx->opcodeParams(pcount);

   for( int32 i = 0; i < pcount; ++ i ) {
      Item* param = params + i;
      if( ! param->isCallable() ) {
         throw new ParamError( ErrorParam(e_inv_params, __LINE__, SRC )
                        .origin( ErrorParam::e_orig_runtime)
                        .extra("C,...") );
      }

      VMContext* nctx = new VMContext(ctx->process(), ctg->m_cg );
      ctg->m_cg->addContext(nctx);

      // copy the callable item in the context,
      // as it will be actually called at launch.
      nctx->pushData(*param);
   }

   return false;
}


static void internal_wait( VMContext* ctx, int pCount, int start, Method* caller, numeric to )
{
   Class* clsShared = Engine::instance()->sharedClass();

   for( int i = start; i < pCount; ++i )
   {
      Item* param = ctx->param(i);
      Class* cls;
      void* inst;
      param->forceClassInst( cls, inst );
      if( !cls->isDerivedFrom( clsShared ) )
      {
         throw caller->paramError();
      }

      Shared* sh = static_cast<Shared*>(cls->getParentData( clsShared, inst ));
      ctx->addWait(sh);
   }

   ctx->engageWait(to);
}

FALCON_DEFINE_METHOD_P( ClassParallel, wait )
{
   if( pCount == 0 )
   {
      throw paramError();
   }

   internal_wait( ctx, pCount, 0, this, -1 );
}


FALCON_DEFINE_METHOD_P( ClassParallel, timedWait )
{
   if( pCount < 2 )
   {
      throw paramError();
   }

   Item* timeout = ctx->param(0);
   if ( ! timeout->isOrdinal() ) {
      throw paramError();
   }

   numeric to = timeout->forceNumeric();
   internal_wait( ctx, pCount, 1, this, to );
}



static void internal_launch( VMContext* ctx, int pCount )
{
   CGCarrier* cgc = static_cast<CGCarrier*>(ctx->self().asInst());
   ContextGroup* cg = cgc->m_cg;
   uint32 count = cg->getContextCount();
   Process* prc = ctx->process();

   for( uint32 i = 0; i < count; ++i ) {
      // call the item at top of each context stack.
      VMContext* nctx = cg->getContext(i);
      Item& toBeCalled = nctx->topData();

      // ... using our parameters.
      for( int32 p = 0; p < pCount; ++p ) {
         nctx->pushData( *ctx->param(p) );
      }

      nctx->callItem( toBeCalled, pCount, 0 );
      //TODO: should we add the whole group?
      prc->addReadyContext( nctx );
   }

}

FALCON_DEFINE_METHOD_P( ClassParallel, launch )
{
   internal_launch( ctx, pCount );

   CGCarrier* cgc = static_cast<CGCarrier*>(ctx->self().asInst());
   cgc->m_cg->configure(ctx->vm(), ctx);

   // and now we wait...
   ctx->addWait( cgc->m_cg->terminated() );
   ctx->engageWait(-1);
   // push a return frame pstep for when we're awaken again.
   ctx->pushReturn();
}


FALCON_DEFINE_METHOD_P( ClassParallel, launchWithResults )
{
   internal_launch( ctx, pCount );

   CGCarrier* cgc = static_cast<CGCarrier*>(ctx->self().asInst());
   cgc->m_cg->configure(ctx->vm(), ctx);

   // and now we wait...
   ctx->addWait( cgc->m_cg->terminated() );
   ClassParallel* cpl = static_cast<ClassParallel*>(methodOf());

   if( !ctx->engageWait(-1) ) {
     // setup a pstep to receive the group result
     ctx->pushCode( &cpl->m_getResult );
   }
   else {
     // we're lucky, all the group is terminated by now.
      cpl->m_getResult.apply( &cpl->m_getResult, ctx );
   }
}



void ClassParallel::PStepGetResults::apply_( const PStep*, VMContext* ctx )
{
   CGCarrier* cgc = static_cast<CGCarrier*>(ctx->self().asInst());
   ItemArray* r = cgc->m_cg->results();
   ctx->returnFrame( r );
}


ClassParallel::PStepGetResults::PStepGetResults()
{
   apply = apply_;
}

}
}

/* end of parallel.cpp */
