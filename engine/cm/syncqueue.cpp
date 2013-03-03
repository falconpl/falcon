/*
   FALCON - The Falcon Programming Language.
   FILE: syncqueue.cpp

   Falcon core module -- Syncronized queue shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/cm/syncqueue.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/syncqueue.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/vm.h>
#include <falcon/stdhandlers.h>

#include <falcon/errors/accesserror.h>

#include <deque>

namespace Falcon {
namespace Ext {

class SharedSyncQueue::Private
{
public:

   typedef std::deque<Item> ValueList;
   ValueList m_values;
   // to invalidate iterators during the gc loop.
   uint32 m_version;
   uint32 m_gcMark;

   Private():
      m_version(0)
   {}

   ~Private() {}
};

SharedSyncQueue::SharedSyncQueue( ContextManager* mgr, const Class* owner ):
   Shared( mgr, owner, false, 0 )
{
   _p = new Private;
   m_fair = false;
   m_held = false;
}

SharedSyncQueue::SharedSyncQueue( ContextManager* mgr, const Class* owner, bool fair ):
   Shared( mgr, owner, fair, 0 )
{
   _p = new Private;
   m_fair = fair;
   m_held = false;
}


SharedSyncQueue::~SharedSyncQueue()
{
   delete _p;
}


int32 SharedSyncQueue::consumeSignal( VMContext*, int32 )
{
   lockSignals();
   int32 result = _p->m_values.empty() ?  0 : 1;
   unlockSignals();

   return result;
}

int32 SharedSyncQueue::lockedConsumeSignal( VMContext*, int32 )
{
   return _p->m_values.empty() ?  0 : 1;
}

void SharedSyncQueue::push( const Item& itm )
{
   lockSignals();
   if( _p->m_values.empty() && ! m_held )
   {
      Shared::lockedSignal(1);
   }
   _p->m_values.push_back(itm);
   _p->m_version++;
   unlockSignals();
}


bool SharedSyncQueue::pop( Item& target )
{
   bool result = false;
   lockSignals();
   if( ! _p->m_values.empty() )
   {
      target = _p->m_values.front();
      _p->m_values.pop_front();
      _p->m_version++;
      result = true;
      if( _p->m_values.empty() && ! m_held )
      {
         Shared::lockedConsumeSignal(0, 1);
      }
   }
   unlockSignals();

   return result;
}


bool SharedSyncQueue::empty() const
{
   lockSignals();
   bool isEmpty = _p->m_values.empty();
   unlockSignals();

   return isEmpty;
}


void SharedSyncQueue::gcMark( uint32 mark )
{
   if( _p->m_gcMark == mark )
   {
      return;
   }

   lockSignals();
   _p->m_gcMark = mark;
   Private::ValueList::iterator iter;
   Private::ValueList::iterator end;
   do
   {
      iter = _p->m_values.begin();
      end = _p->m_values.end();

      while( iter != end )
      {
         Item current = *iter;
         uint32 version = _p->m_version;
         unlockSignals();

         current.gcMark(mark);

         lockSignals();
         if ( version != _p->m_version )
         {
            // try again.
            break;
         }
         ++iter;
      }

   } while( iter != end );
   unlockSignals();
}


uint32 SharedSyncQueue::currentMark() const
{
   return _p->m_gcMark;
}


//=============================================================
//
//

FairSyncQueue::FairSyncQueue( ContextManager* mgr, const Class* owner ):
         SharedSyncQueue(mgr, owner, true)
{
}

FairSyncQueue::~FairSyncQueue()
{
}


int32 FairSyncQueue::consumeSignal( VMContext* ctx, int32 count )
{
   lockSignals();
   if( !m_held && Shared::lockedConsumeSignal(ctx, count) > 0 )
   {
      m_held = true;
      unlockSignals();
      return 1;
   }

   unlockSignals();
   return 0;
}


void FairSyncQueue::signal( int32 )
{
   lockSignals();
   m_held = false;
   if( !_p->m_values.empty() )
   {
      Shared::lockedSignal(1);
   }
   unlockSignals();
}

int32 FairSyncQueue::lockedConsumeSignal( VMContext* ctx, int32 count )
{
   if( !m_held && Shared::lockedConsumeSignal(ctx, count) > 0 )
   {
      m_held = true;
      return 1;
   }
   return 0;
}

//=============================================================
//

ClassSyncQueue::ClassSyncQueue():
      ClassShared("SyncQueue"),
      FALCON_INIT_PROPERTY(empty),

      FALCON_INIT_METHOD(push),
      FALCON_INIT_METHOD(pop),
      FALCON_INIT_METHOD(wait)
{
   static Class* shared = Engine::handlers()->sharedClass();
   addParent(shared);
}

ClassSyncQueue::~ClassSyncQueue()
{}

void* ClassSyncQueue::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassSyncQueue::op_init( VMContext* ctx, void*, int pCount ) const
{
   Shared* shared;
   if( pCount == 0 || ! ctx->opcodeParams(pCount)->isTrue() )
   {
      shared = new SharedSyncQueue(&ctx->vm()->contextManager(), this);
   }
   else {
      shared = new FairSyncQueue(&ctx->vm()->contextManager(), this);
   }

   ctx->stackResult(pCount+1, FALCON_GC_HANDLE(shared));
   return true;
}



FALCON_DEFINE_PROPERTY_GET_P( ClassSyncQueue, empty )
{
   SharedSyncQueue* sc = static_cast<SharedSyncQueue*>(instance);
   value.setBoolean( sc->empty() );
}


FALCON_DEFINE_PROPERTY_SET( ClassSyncQueue, empty )( void*, const Item& )
{
   throw readOnlyError();
}


FALCON_DEFINE_METHOD_P( ClassSyncQueue, push )
{
   if( pCount == 0 )
   {
      throw paramError( __LINE__, SRC );
   }

   SharedSyncQueue* queue = static_cast<SharedSyncQueue*>(ctx->self().asInst());
   for( int32 i = 0; i < pCount; ++i )
   {
      queue->push( *ctx->param(i) );
   }

   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassSyncQueue, pop )
{
   SharedSyncQueue* queue = static_cast<SharedSyncQueue*>(ctx->self().asInst());

   Item dflt;

   if( queue->isFair() )
   {
      if( ctx->acquired() != queue )
      {
         throw FALCON_SIGN_XERROR( AccessError, e_acc_forbidden, .extra("Cannot access pop() if not owner") );
      }
      else {
         queue->pop( dflt );
         ctx->releaseAcquired();
      }
   }
   else {
      // check and get params.
      Item* i_dflt = ctx->param(0);
      if( i_dflt != 0 )
      {
         dflt = *i_dflt;
         dflt.copied(true);
      }

      queue->pop( dflt );
   }

   ctx->returnFrame(dflt);
}

FALCON_DEFINE_METHOD_P( ClassSyncQueue, wait )
{
   static const PStep& stepWaitSuccess = Engine::instance()->stdSteps()->m_waitSuccess;
   static const PStep& stepInvoke = Engine::instance()->stdSteps()->m_reinvoke;

   if( ctx->releaseAcquired() )
   {
      ctx->pushCode( &stepInvoke );
      return;
   }

   //===============================================
   //
   int64 timeout = -1;
   if( pCount >= 1 )
   {
      Item* i_timeout = ctx->param(0);
      if (!i_timeout->isOrdinal())
      {
         throw paramError(__LINE__, SRC);
      }

      timeout = i_timeout->forceInteger();
   }

   Shared* shared = static_cast<Shared*>(ctx->self().asInst());
   ctx->initWait();
   ctx->addWait(shared);
   shared = ctx->engageWait( timeout );

   if( shared != 0 )
   {
      ctx->returnFrame( Item().setBoolean(true) );
   }
   else {
      // we got to wait.
      ctx->pushCode( &stepWaitSuccess );
   }
}

}
}

/* end of syncqueue.cpp */
