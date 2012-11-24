/*
   FALCON - The Falcon Programming Language.
   FILE: syncqueue_posix.h

   Synchronous queue template class -- POSIX specialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 04 Nov 2012 18:32:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYNCQUEUE_POSIX_H_
#define _FALCON_SYNCQUEUE_POSIX_H_

#include <deque>
#include <pthread.h>
#include <sys/time.h>
#include <falcon/fassert.h>
#include <errno.h>

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
      pthread_cond_init(&m_filled, NULL);
      pthread_mutex_init(&m_mtx, NULL);
   }
   ~SyncQueue() {
      pthread_cond_destroy(&m_filled);
      pthread_mutex_destroy(&m_mtx);
   }

   void add( __T data ) {
      pthread_mutex_lock(&m_mtx);
      m_queue.push_back(data);
      pthread_mutex_unlock(&m_mtx);
      pthread_cond_broadcast(&m_filled);
   }

   bool get( __T& data, int *terminated ) {
      pthread_mutex_lock(&m_mtx);
      while( m_queue.empty() && ! m_terminateWaiters && ! *terminated) {
         pthread_cond_wait( &m_filled, &m_mtx );
      }
      if (m_terminateWaiters || *terminated ) {
         *terminated = 1;
         pthread_mutex_unlock( &m_mtx );
         return false;
      }
      data = m_queue.front();
      m_queue.pop_front();
      pthread_mutex_unlock(&m_mtx);

      return true;
   }

   void terminateWaiters() {
      pthread_mutex_lock(&m_mtx);
      m_terminateWaiters = true;
      pthread_mutex_unlock(&m_mtx);
      pthread_cond_broadcast(&m_filled);
   }

   bool tryGet( __T& data, int *terminated ) {
      pthread_mutex_lock(&m_mtx);
      if( m_terminateWaiters )
      {
         *terminated = 1;
         pthread_mutex_unlock(&m_mtx);
         return false;
      }
      if( ! m_queue.empty()  ) {
         data = m_queue.front();
         m_queue.pop_front();
         pthread_mutex_unlock(&m_mtx);
         return true;
      }
      else {
         pthread_mutex_unlock(&m_mtx);
         return false;
      }
   }

   bool getTimed( __T& data, int to, int *terminated ) {
      struct timespec timeToWait;
      struct timeval now;
      int rt;

      gettimeofday(&now,NULL);
      timeToWait.tv_sec = now.tv_sec + (to/1000);
      timeToWait.tv_nsec = (now.tv_usec + (to%1000) * 1000) * 1000;

      pthread_mutex_lock(&m_mtx);
      while( m_queue.empty() && ! m_terminateWaiters && ! *terminated) {
         rt = pthread_cond_timedwait( &m_filled, &m_mtx, &timeToWait );
         if( rt == ETIMEDOUT )
         {
            //pthread_mutex_unlock( &m_mtx );
            if (m_terminateWaiters) {
               *terminated = 1;
            }
            return false;
         }
         fassert2( rt == 0, "Error waiting for the condition variable");
      }

      if (m_terminateWaiters || *terminated ) {
         *terminated = 1;
         pthread_mutex_unlock( &m_mtx );
         return false;
      }
      data = m_queue.front();
      m_queue.pop_front();
      pthread_mutex_unlock(&m_mtx);

      return true;
   }

   bool isTerminated() {
      pthread_mutex_lock(&m_mtx);
      bool term = m_terminateWaiters;
      pthread_mutex_unlock(&m_mtx);
      return term;
   }

   void terminateOne( int* termHandle ) {
      pthread_mutex_lock(&m_mtx);
      termHandle = 1;
      pthread_mutex_unlock(&m_mtx);
      pthread_cond_broadcast(&m_filled);

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
   pthread_mutex_t m_mtx;
   pthread_cond_t m_filled;

   bool m_terminateWaiters;
};
}

#endif

/* end of syncqueue_posix.h */
