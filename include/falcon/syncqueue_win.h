/*
   FALCON - The Falcon Programming Language.
   FILE: syncqueue_win.h

   Synchronous queue template class -- Windows Event based specialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai, Paul Davey
   Begin: Sun, 04 Nov 2012 18:32:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYNCQUEUE_WIN_H_
#define _FALCON_SYNCQUEUE_WIN_H_

#include <deque>
#include <falcon/fassert.h>
#include <windows.h>

namespace Falcon
{

/**
 Utility class for synchronous wait queues.

 This is used by the engine to exchange messages between known agents,
 as the context manager and the processors.

 */
template<class __T>
class SyncQueue
{
public:
   SyncQueue() {
      m_terminateWaiters = false;
      m_filled = CreateEvent(NULL, TRUE, FALSE, NULL);
      InitializeCriticalSectionAndSpinCount(&m_mtx,500);
   }
   ~SyncQueue() {
      CloseHandle(m_filled);
      DeleteCriticalSection(&m_mtx);
   }

   void add( __T data ) {
      EnterCriticalSection(&m_mtx);
      m_queue.push_back(data);
      SetEvent(m_filled);
      LeaveCriticalSection(&m_mtx);
   }

   bool get( __T& data, int *terminated ) {
      EnterCriticalSection(&m_mtx);
      while( m_queue.empty() && ! m_terminateWaiters && ! *terminated) {
         LeaveCriticalSection(&m_mtx);
         WaitForSingleObject( m_filled, INFINITE );
         EnterCriticalSection(&m_mtx);
      }
      if (m_terminateWaiters || *terminated ) {
         *terminated = 1;
         LeaveCriticalSection(&m_mtx);
         return false;
      }
      data = m_queue.front();
      m_queue.pop_front();
      if (m_queue.empty())
      {
         ResetEvent(m_filled);
      }
      LeaveCriticalSection(&m_mtx);

      return true;
   }

   void terminateWaiters() {
      EnterCriticalSection(&m_mtx);
      m_terminateWaiters = true;
      SetEvent(m_filled);
      LeaveCriticalSection(&m_mtx);
   }

   bool tryGet( __T& data, int *terminated ) {
      EnterCriticalSection(&m_mtx);
      if( m_terminateWaiters )
      {
         *terminated = 1;
         LeaveCriticalSection(&m_mtx);
         return false;
      }
      if( ! m_queue.empty()  ) {
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

   bool getTimed( __T& data, int to, int *terminated ) {
      int rt;

      EnterCriticalSection(&m_mtx);
      while( m_queue.empty() && ! m_terminateWaiters && ! *terminated) {
         LeaveCriticalSection(&m_mtx);
         rt = WaitForSingleObject( m_filled, to );
         EnterCriticalSection(&m_mtx);
         if( rt == WAIT_TIMEOUT )
         {
            if (m_terminateWaiters) {
               *terminated = 1;
            }
            LeaveCriticalSection(&m_mtx);
            return false;
         }
         fassert2( rt == 0, "Error waiting for the condition variable");
      }

      if (m_terminateWaiters || *terminated ) {
         *terminated = 1;
         LeaveCriticalSection(&m_mtx);
         return false;
      }
      data = m_queue.front();
      m_queue.pop_front();
      if (m_queue.empty())
      {
         ResetEvent(m_filled);
      }
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
      SetEvent(m_filled);
      LeaveCriticalSection(&m_mtx);
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
   HANDLE m_filled;

   bool m_terminateWaiters;
};
}

#endif

/* end of syncqueue_win.h */
