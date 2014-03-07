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
#include <falcon/itemarray.h>
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>
#include <falcon/function.h>

#include <falcon/stderrors.h>

namespace Falcon {
namespace Ext {

   
class CGCarrier
{
public:
   ContextGroup* m_cg;
   uint32 m_mark;

   CGCarrier(ContextGroup* grp ):
      m_mark(0)
   {
      m_cg = grp;
   }

   CGCarrier( const CGCarrier& other ):
      m_cg( other.m_cg ),
      m_mark(0)
   {
      m_cg->incref();
   }

   virtual ~CGCarrier()
   {
      m_cg->decref();
   }

   virtual CGCarrier* clone() const { return new CGCarrier(*this); }
};


//====================================================================
//
//

namespace CParallel  {

FALCON_DECLARE_FUNCTION( wait, "..." );
FALCON_DECLARE_FUNCTION( tryWait, "..." );
FALCON_DECLARE_FUNCTION( timedWait, "timeout:N,..." );
FALCON_DECLARE_FUNCTION( add, "C..." );
FALCON_DECLARE_FUNCTION( launch, "..." );
FALCON_DECLARE_FUNCTION( launchWithResults, "..." );



static void internal_wait( VMContext* ctx, int pCount, int start, Function* caller, int64 to )
{
   static Class* clsShared = Engine::handlers()->sharedClass();
   static PStep* step = &Engine::instance()->stdSteps()->m_waitComplete;
   static PStep* stepInvoke = &Engine::instance()->stdSteps()->m_reinvoke;

   // return if we have pending signals -- we'll be called back.
   if( ctx->releaseAcquired())
   {
      ctx->pushCode(stepInvoke);
      return;
   }

   ctx->initWait();
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

   Shared* sh = ctx->engageWait(to);
   if( sh != 0 )
   {
      // return the resource.
      ctx->returnFrame(Item(sh->handler(), sh));
   }
   else if( to  == 0 )
   {
      // return nil immediately
      ctx->returnFrame();
   }
   else {
      // try later.
      ctx->pushCode( step );
   }
}


void Function_wait::invoke(VMContext* ctx, int32 pCount )
{
   if( pCount == 0 )
   {
      throw paramError();
   }

   internal_wait( ctx, ctx->paramCount(), 0, this, -1 );
}


void Function_tryWait::invoke(VMContext* ctx, int32 pCount )
{
   if( pCount == 0 )
   {
      throw paramError();
   }

   internal_wait( ctx, ctx->paramCount(), 0, this, 0);
}



void Function_timedWait::invoke(VMContext* ctx, int32 pCount )
{
   if( pCount < 2 )
   {
      throw paramError();
   }

   Item* timeout = ctx->param(0);
   if ( ! timeout->isOrdinal() ) {
      throw paramError();
   }

   int64 to = timeout->forceInteger();
   internal_wait( ctx, ctx->paramCount(), 1, this, to );
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

      ctx->callerLine(__LINE__+1);
      nctx->callItem( toBeCalled, pCount, 0 );
      //TODO: should we add the whole group?
      prc->startContext( nctx );
   }
}


void Function_add::invoke(VMContext* ctx, int32 pCount )
{
   if ( pCount == 0 )
   {
      throw new ParamError( ErrorParam(e_inv_params, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime )
               .extra("C,...") );
   }

   CGCarrier* ctg = static_cast<CGCarrier*>(ctx->self().asInst());

   for( int32 i = 0; i < pCount; ++ i ) {
      Item* param = ctx->param(i);
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

   ctx->returnFrame(ctx->self());
}


void Function_launch::invoke(VMContext* ctx, int32 pCount )
{
   internal_launch( ctx, pCount );

   CGCarrier* cgc = static_cast<CGCarrier*>(ctx->self().asInst());
   cgc->m_cg->configure(ctx->vm(), ctx);

   // and now we wait...
   ctx->addWait( cgc->m_cg->terminated() );
   ctx->engageWait(-1);

   // push a return frame pstep for when we're awaken again.
   ClassParallel* cpl = static_cast<ClassParallel*>(methodOf());
   ctx->pushCode( &cpl->m_afterWait );
}


void Function_launchWithResults::invoke(VMContext* ctx, int32 pCount )
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

}

//===============================================================================
//


ClassParallel::ClassParallel():
   Class("Parallel")
{
   addMethod( new CParallel::Function_wait, true );
   addMethod( new CParallel::Function_tryWait, true );
   addMethod( new CParallel::Function_timedWait, true );

   addMethod( new CParallel::Function_add );
   addMethod( new CParallel::Function_launch );
   addMethod( new CParallel::Function_launchWithResults );
}

ClassParallel::~ClassParallel()
{}


void* ClassParallel::createInstance() const
{
   CGCarrier* ctg = new CGCarrier(new ContextGroup );
   return ctg;
}

void ClassParallel::dispose( void* instance ) const
{
   CGCarrier* ctg = static_cast<CGCarrier*>(instance);
   delete ctg;
}

void* ClassParallel::clone( void* instance ) const
{
   CGCarrier* ctg = static_cast<CGCarrier*>(instance);
   return ctg->clone();
}

void ClassParallel::gcMarkInstance( void* instance, uint32 mark ) const
{
   CGCarrier* ctg = static_cast<CGCarrier*>(instance);
   ctg->m_mark = mark;
}

bool ClassParallel::gcCheckInstance( void* instance, uint32 mark ) const
{
   CGCarrier* ctg = static_cast<CGCarrier*>(instance);
   return ctg->m_mark >= mark;
}


bool ClassParallel::op_init( VMContext* ctx, void* instance, int pcount ) const
{
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


void ClassParallel::PStepGetResults::apply_( const PStep*, VMContext* ctx )
{
   CGCarrier* cgc = static_cast<CGCarrier*>(ctx->self().asInst());
   Error* err = cgc->m_cg->error();
   if( err ) {
      ctx->raiseError(err);
   }
   else  {
      ItemArray* r = cgc->m_cg->results();
      ctx->returnFrame( FALCON_GC_HANDLE( r ) );
   }
}


ClassParallel::PStepGetResults::PStepGetResults()
{
   apply = apply_;
}


void ClassParallel::PStepAfterWait::apply_( const PStep*, VMContext* ctx )
{
   CGCarrier* cgc = static_cast<CGCarrier*>(ctx->self().asInst());
   Error* err = cgc->m_cg->error();
   if( err ) {
      ctx->raiseError(err);
   }
   else {
      ctx->returnFrame();
   }
}

ClassParallel::PStepAfterWait::PStepAfterWait()
{
   apply = apply_;
}

}
}

/* end of parallel.cpp */
