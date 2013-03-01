/*
   FALCON - The Falcon Programming Language.
   FILE: shared.cpp

   Shared resource
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 29 Jul 2012 16:25:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#define SRC "engine/shared.cpp"

#include <falcon/shared.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/contextmanager.h>

#include "shared_private.h"

namespace Falcon
{

Shared::Shared( ContextManager* mgr, const Class* handler, bool acquireable, int32 signals ):
   m_mark(0),
   m_notifyTo(mgr),
   m_acquireable( acquireable ),
   m_cls( handler ),
   m_refcounter_Shared(1)
{
   _p = new Private( signals );
}

Shared::~Shared() {
   delete _p;
}

Shared* Shared::clone() const
{
   return new Shared( m_notifyTo, m_cls, m_acquireable, _p->m_signals );
}

int32 Shared::consumeSignal( VMContext*, int32 count )
{
   _p->m_mtx.lock();
   if( count > _p->m_signals ) {
      count = _p->m_signals;
   }
   _p->m_signals -= count;
   _p->m_mtx.unlock();

   return count;
}


int Shared::lockedConsumeSignal( VMContext*, int count )
{
   if( count > _p->m_signals ) {
      count = _p->m_signals;
   }
   _p->m_signals -= count;

   return count;
}

void Shared::signal( int32 count )
{
   if( count <= 0 )
   {
      return;
   }

   _p->m_mtx.lock();
   bool doSignal = (_p->m_signals == 0 );
   _p->m_signals += count;
   _p->m_mtx.unlock();

   if( doSignal )
   {
      m_notifyTo->onSharedSignaled(this);
   }
}

void Shared::lockedSignal( int32 count )
{
   bool doSignal = (_p->m_signals == 0 );
   _p->m_signals += count;
   if( doSignal )
   {
      m_notifyTo->onSharedSignaled(this);
   }
}

void Shared::lockSignals() const
{
   _p->m_mtx.lock();
}

void Shared::unlockSignals() const
{
   _p->m_mtx.unlock();
}


void Shared::onWaiterWaiting(VMContext*)
{
   // nothing to do
}


int32 Shared::signalCount() const
{
   _p->m_mtx.lock();
   int32 count = _p->m_signals;
   _p->m_mtx.unlock();

   return count;
}

int32 Shared::lockedSignalCount() const
{
   return _p->m_signals;
}


void Shared::broadcast()
{
   ContextManager* notifyTo = 0;
   _p->m_mtx.lock();
   int32 count= (int32) _p->m_waiters.size();
   if( count > 0 )
   {
      notifyTo = &_p->m_waiters.front()->vm()->contextManager();
      _p->m_signals += count;
   }
   _p->m_mtx.unlock();

   if( notifyTo != 0 ) {
      notifyTo->onSharedSignaled(this);
   }
}

void Shared::dropWaiting( VMContext* ctx )
{
   _p->m_mtx.lock();
   Private::ContextList::iterator iter = _p->m_waiters.begin();
   while( iter != _p->m_waiters.end() ) {
      VMContext* wctx = *iter;
      if( ctx == wctx ) {
         ctx->decref();
         _p->m_waiters.erase(iter);
         break;
      }
      ++iter;
   }
   _p->m_mtx.unlock();
}


bool Shared::addWaiter( VMContext* ctx )
{
   _p->m_mtx.lock();
   if ( lockedConsumeSignal(ctx, 1) )
   {
      _p->m_mtx.unlock();

      if( m_acquireable )
      {
         ctx->acquire(this);
      }
      ctx->signaledResource(this);

      return true;
   }

   _p->m_waiters.push_back(ctx);
   ctx->incref();
   _p->m_mtx.unlock();
   return false;
}


void Shared::onWakeupComplete()
{
}


}

/* end of shared.cpp */
