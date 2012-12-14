/*
   FALCON - The Falcon Programming Language.
   FILE: syncqueue_win6.h

   Synchronous queue template class -- Windows vista specialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 04 Nov 2012 18:32:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYNCQUEUE_WIN6_H_
#define _FALCON_SYNCQUEUE_WIN6_H_

#include <windows.h>
#include <falcon/fassert.h>

#include <deque>

namespace Falcon
{

/**
 Utility class for synchronous wait queues.

 This is used by the engine to exchange messages between known agents,
 as the context manager and the processors.

 */
template<class __T>
class SyncQueue {
   SyncQueue() {
      m_terminateWaiters = false;
      InitializeCriticalSectionAndSpinCount(&m_mtx,500);
      InitializeConditionVariable(&m_filled);
   }
   ~SyncQueue() {
      DeleteConditionVariable(&m_filled);
      DeleteCriticalSection(&m_mtx);
   }

   void add( __T data ) {
      EnterCriticalSection(&m_mtx);
      m_queue.push_back(data);
      pthread_mutex_unlock(&m_mtx);
      LeaveCriticalSection(&m_filled);
   }

   bool get( __T& data, int* terminated ) {
      EnterCriticalSection(&m_mtx);
      while( m_queue.empty() && ! m_terminateWaiters && ! *terminated) {
         pthread_cond_wait( &m_filled, &m_mtx );
      }
      if (m_terminateWaiters || *terminated) {
         *terminated = 1;
         LeaveCriticalSection( &m_mtx );
         return false;
      }
      data = m_queue.front();
      m_queue.pop_front();
      LeaveCriticalSection(&m_mtx);

      return true;
   }

   void terminateWaiters() {
      EnterCriticalSection(&m_mtx);
      m_terminateWaiters = true;
      LeaveCriticalSection(&m_mtx);
      WakeAllConditionVariable(&m_filled);
   }

   bool tryGet( __T& data, int* terminated ) {
      EnterCriticalSection(&m_mtx);
      if(m_terminateWaiters ) {
         *terminated = 1;
         LeaveCriticalSection(&m_mtx);
         return false;
      }
      if( ! m_queue.empty() ) {
         data = m_queue.front();
         m_queue.pop_front();
         LeaveCriticalSection(&m_mtx);
         return true;
      }
      else {
         LeaveCriticalSection(&m_mtx);
         return false;
      }
   }

   bool getTimed( __T& data, int to, int* terminated ) {
      int rt;

      EnterCriticalSection(&m_mtx);
      while( m_queue.empty() && ! m_terminateWaiters ) {
         rt = SleepConditionVariableCS( &m_filled, &m_mtx, &timeToWait );
         if( rt == 0 && GetLastError() == ERROR_TIMEOUT )
         {
            if( m_terminateWaiters ) {
               *terminated = 1;
            }
            LeaveCriticalSection( &m_mtx );
            return false;
         }
         fassert2( rt != 0, "Error waiting for the condition variable");
      }

      if (m_terminateWaiters || *terminated ) {
         *terminated = 1;
         LeaveCriticalSection( &m_mtx );
         return false;
      }
      data = m_queue.front();
      m_queue.pop_front();
      LeaveCriticalSection(&m_mtx);

      return true;
   }

   bool isTerminated() {
      EnterCriticalSection(&m_mtx);
      bool term = m_terminateWaiters;
      LeaveCriticalSection(&m_mtx);
      return term;
   }

   void terminateOne( int* termHandle ) {
      EnterCriticalSection(&m_mtx);
      *termHandle = 1;
      EnterCriticalSection(&m_mtx);
      WakeAllConditionVariable(&m_filled);
   }

   bool getST( __T& data ) {
      if ( m_queue.empty() ) {
         return false;
      }
      data = m_queue.front();
      m_queue.pop_front();
      return true;
   }

private:
   std::deque<__T> m_queue;
   CRITICAL_SECTION m_mtx;
   CONDITION_VARIABLE m_filled;

   bool m_terminateWaiters;
};
}

#endif

/* end of syncqueue_win6.h */
