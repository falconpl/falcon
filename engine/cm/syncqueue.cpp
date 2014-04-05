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
#include <falcon/stderrors.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/vm.h>
#include <falcon/stdhandlers.h>
#include <falcon/function.h>

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
   bool done = false;
   while( ! done )
   {
      done = true;
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
            done = false;
            break;
         }
         ++iter;
      }

   }
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
//

/*#
  @property empty SyncQueue
  @brief Checks if queue is empty at the moment.

  This information has a meaning only if it can be demonstrated
  that there aren't other producers able to push data in the queue
  in this moment.

 */
static void get_empty( const Class*, const String&, void *instance, Item& value )
{
   SharedSyncQueue* sc = static_cast<SharedSyncQueue*>(instance);
   value.setBoolean( sc->empty() );
}

namespace CSyncQueue {

/*#
  @method push SyncQueue
  @brief Pushes one or more items in the queue.
  @param item The item pushed in the queue.
  @optparam ... More items to be pushed atomically.

  It is not necessary to acquire the queue to push an item.
  Also, pushing an item does not automatically release the queue.
 */
FALCON_DECLARE_FUNCTION( push, "item:X,..." );

/*#
  @method pop SyncQueue
  @brief Removes an item from the queue atomically, or waits for an item to be available.
  @optparam onEmpty Returned if the queue is empty.
  @raise AccessError if the queue is in fair mode and the pop method is invoked without
        having acquired the resource with a successfull wait.

   In non fair mode, even if the queue is signaled and the wait operation is successful,
   there is no guarantee that the queue is still non-empty when this agent
   tires to pop the queue. The pop method is granted to return an item from
   the queue if and only if a wait operation was successful and there aren't other
   agents trying to pop from this resource.

   In fair mode, this method can be invoked only after having acquired the queue
   through a successful wait operation. It is then granted that the method will return
   an item, and the @b onEmpty parameter, if given, will be ignored.
 */
FALCON_DECLARE_FUNCTION( pop, "onEmpty:[X]" );

/*#
  @method wait SyncQueue
  @brief Wait the queue to be non-empty.
  @optparam timeout Milliseconds to wait for the barrier to be open.
  @return true if the barrier is open during the wait, false if the given timeout expires.

  If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
  the wait exits immediately.

  If the queue is in fair mode, a successful wait makes the invoker to
  enter a critical section; the pop method will then release the queue.
 */
FALCON_DECLARE_FUNCTION( wait, "timeout:[N]" );


void Function_push::invoke( VMContext* ctx, int32 pCount )
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


void Function_pop::invoke( VMContext* ctx, int32 )
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
      }

      queue->pop( dflt );
   }

   ctx->returnFrame(dflt);
}


void Function_wait::invoke( VMContext* ctx, int32 pCount )
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

//=============================================================
//
//

ClassSyncQueue::ClassSyncQueue():
      ClassShared("SyncQueue")
{
   static Class* shared = Engine::handlers()->sharedClass();
   setParent(shared);

   addProperty("empty", &get_empty);

   addMethod( new CSyncQueue::Function_push);
   addMethod( new CSyncQueue::Function_pop);
   addMethod( new CSyncQueue::Function_wait);
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

}
}

/* end of syncqueue.cpp */
