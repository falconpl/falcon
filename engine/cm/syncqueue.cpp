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

#include <deque>

namespace Falcon {
namespace Ext {

class SharedSyncQueue::Private
{
public:

   typedef std::deque<Item> ValueList;
   ValueList m_values;
   Mutex m_mtx_content;
   // to invalidate iterators during the gc loop.
   uint32 m_version;
   bool m_bHeld;
   uint32 m_gcMark;

   Private():
      m_version(0),
      m_bHeld(false)
   {}

   ~Private() {}
};

SharedSyncQueue::SharedSyncQueue( const Class* owner ):
   Shared(owner, true, 0 )
{
   _p = new Private;
}

SharedSyncQueue::~SharedSyncQueue()
{
   delete _p;
}


void SharedSyncQueue::signal( int32 )
{
   release();
}

int32 SharedSyncQueue::consumeSignal( int32 )
{
   _p->m_mtx_content.lock();
   if( ! _p->m_bHeld && ! _p->m_values.empty() )
   {
      _p->m_bHeld = true;
      _p->m_mtx_content.unlock();

      Shared::consumeSignal(1);
      return 1;
   }

   _p->m_mtx_content.unlock();
   return 0;
}

bool SharedSyncQueue::lockedConsumeSignal()
{
   _p->m_mtx_content.lock();
   if( ! _p->m_bHeld && ! _p->m_values.empty() )
   {
      _p->m_bHeld = true;
      _p->m_mtx_content.unlock();

      Shared::lockedConsumeSignal();
      return true;
   }

   _p->m_mtx_content.unlock();
   return false;
}

void SharedSyncQueue::push( const Item& itm )
{
   bool doSignal = false;

   _p->m_mtx_content.lock();
   _p->m_values.push_back(itm);
   _p->m_version++;
   if( ! _p->m_bHeld )
   {
      doSignal = true;
   }
   _p->m_mtx_content.unlock();

   if( doSignal )
   {
      Shared::signal(1);
   }
}

bool SharedSyncQueue::pop( Item& target )
{
   bool result = false;

   _p->m_mtx_content.lock();
   if( ! _p->m_bHeld && ! _p->m_values.empty() )
   {
      target = _p->m_values.front();
      _p->m_values.pop_front();
      _p->m_version++;
      result = true;

      if( _p->m_values.empty() )
      {
         _p->m_mtx_content.unlock();
         Shared::consumeSignal(1);
      }
      else {
         _p->m_mtx_content.unlock();
      }
   }
   else {
      _p->m_mtx_content.unlock();
   }

   return result;
}


bool SharedSyncQueue::empty() const
{
   _p->m_mtx_content.lock();
   bool isEmpty = _p->m_values.empty();
   _p->m_mtx_content.unlock();

   return isEmpty;
}

void SharedSyncQueue::release()
{
   _p->m_mtx_content.lock();
   if( _p->m_bHeld )
   {
      _p->m_bHeld = false;
      if( ! _p->m_values.empty() )
      {
         _p->m_mtx_content.unlock();
         Shared::signal(1);
      }
      else {
         // nothing to signal.
         _p->m_mtx_content.unlock();
      }
   }
   else {
      _p->m_mtx_content.unlock();
   }
}


void SharedSyncQueue::gcMark( uint32 mark )
{
   if( _p->m_gcMark == mark )
   {
      return;
   }

   _p->m_mtx_content.lock();
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
         _p->m_mtx_content.unlock();

         current.gcMark(mark);

         _p->m_mtx_content.lock();
         if ( version != _p->m_version )
         {
            // try again.
            break;
         }
      }

   } while( iter != end );
   _p->m_mtx_content.unlock();
}


uint32 SharedSyncQueue::currentMark() const
{
   return _p->m_gcMark;
}


//=============================================================
//

ClassSyncQueue::ClassSyncQueue():
      ClassShared("SyncQueue"),
      FALCON_INIT_PROPERTY(empty),

      FALCON_INIT_METHOD(push),
      FALCON_INIT_METHOD(pop),
      FALCON_INIT_METHOD(tryPop),
      FALCON_INIT_METHOD(wait),
      FALCON_INIT_METHOD(release)
{
   static Class* shared = Engine::instance()->sharedClass();
   addParent(shared);
}

ClassSyncQueue::~ClassSyncQueue()
{}

void* ClassSyncQueue::createInstance() const
{
   return new SharedSyncQueue(this);
}


bool ClassSyncQueue::op_init( VMContext*, void*, int ) const
{
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

   // check and get params.
   Item* i_timeout = ctx->param(0);
   Item* i_dflt = ctx->param(1);
   if( i_timeout != 0 && !(i_timeout->isNil() || i_timeout->isOrdinal()) )
   {
      throw paramError(__LINE__, SRC);
   }


   int64 timeout = i_timeout == 0 || i_timeout->isNil() ? -1 : i_timeout->forceInteger();
   Item dflt;
   if( i_dflt != 0 )
   {
      dflt = *i_dflt;
      dflt.copied(true);
   }

   Item target;
   bool result = queue->pop( target );

   // success or not, we abandoned the resource nevertheless.
   if( ctx->acquired() == queue )
   {
      queue->release();
   }
   ctx->releaseAcquired();

   if( result )
   {
      ctx->returnFrame(target);
      return;
   }

   // we failed..
   if( timeout == 0 )
   {
      ctx->returnFrame(dflt);
      return;
   }

   // prepare for wait -- we need a landing pstep.
   ctx->pushCode( &static_cast<ClassSyncQueue*>(methodOf())->m_stepAfterPop );

   ctx->initWait();
   ctx->addWait( queue );
   ctx->engageWait( timeout );
}


FALCON_DEFINE_METHOD_P1( ClassSyncQueue, tryPop )
{
   SharedSyncQueue* queue = static_cast<SharedSyncQueue*>(ctx->self().asInst());

   // check and get params.
   Item* i_dflt = ctx->param(0);
   Item dflt;
   if( i_dflt != 0 )
   {
      dflt = *i_dflt;
      dflt.copied(true);
   }

   queue->pop( dflt );
   ctx->returnFrame(dflt);
}


void ClassSyncQueue::PStepAfterPop::apply_(const PStep*, VMContext* ctx )
{
   SharedSyncQueue* shared = static_cast<SharedSyncQueue*>( ctx->getSignaledResouce() );
   Item value;

   if( shared == 0 || ! shared->pop(value) )
   {
      // we must return the default item.
      Item* i_dflt = ctx->param(1);
      if( i_dflt != 0 )
      {
         ctx->returnFrame(*i_dflt);
      }
      else {
         ctx->returnFrame();
      }
   }
   else {
      ctx->returnFrame(value);
   }

   if( shared != 0 )
   {
      ctx->releaseAcquired();
      shared->release();
   }
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

FALCON_DEFINE_METHOD_P1( ClassSyncQueue, release )
{
   SharedSyncQueue* queue = static_cast<SharedSyncQueue*>(ctx->self().asInst());

   if( ctx->acquired() == queue )
   {
      ctx->releaseAcquired();
      queue->release();
   }

   ctx->returnFrame();
}

}
}

/* end of syncqueue.cpp */
