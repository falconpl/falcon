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
#include "shared_private.h"

namespace Falcon
{

Shared::Shared( Class* handler, bool acquireable, int32 signals ):
   m_cls( handler ),
   m_acquireable( acquireable )
{
   _p = new Private( signals );
}

Shared::~Shared() {
   delete _p;
}

int32 Shared::consumeSignal( int32 count )
{
   _p->m_mtx.lock();
   if( count > _p->m_signals ) {
      count = _p->m_signals;
   }
   _p->m_signals -= count;
   _p->m_mtx.unlock();

   return count;
}

void Shared::signal( int32 count )
{
   Scheduler* notifyTo = 0;

   _p->m_mtx.lock();
   _p->m_signals += count;
   if( ! _p->m_waiters.empty() )
   {
      notifyTo = _p->m_waiters.front()->vm()->scheduler();
   }
   _p->m_mtx.unlock();

   if( notifyTo != 0 ) {
      notifyTo->signalResource(this);
   }
}


void Shared::broadcast()
{
   Scheduler* notifyTo = 0;
   _p->m_mtx.lock();
   int32 count= (int32) _p->m_waiters.size();
   if( count > 0 )
   {
      notifyTo = _p->m_waiters.front()->vm()->scheduler();
      _p->m_signals += count;
   }
   _p->m_mtx.unlock();

   if( notifyTo != 0 ) {
      notifyTo->broadcastResource(this);
   }
}

void Shared::dropWaiting( VMContext* ctx )
{
   _p->m_mtx.lock();
   Private::ContextList::iterator iter = _p->m_waiters.begin();
   while( iter != _p->m_waiters.end() ) {
      VMContext* wctx = *iter;
      if( ctx == wctx ) {
         _p->m_waiters.erase(iter);
         break;
      }
      ++iter;
   }
   _p->m_mtx.unlock();
}

}

/* end of shared.cpp */
