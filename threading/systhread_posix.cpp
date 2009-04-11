/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: systhread_posix.cpp

   System dependent MT provider - posix specific.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Apr 2008 21:32:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependent MT provider - posix specific.
*/

#include <falcon/memory.h>
#include "./mt.h"
#include "systhread.h"
#include "systhread_posix.h"

#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <errno.h>

//#define TRACE_ON

#ifdef TRACE_ON
   #include <stdio.h>
   #define TRACE printf
#else
   inline void _trace( const char* fmt, ...) { }
   #define TRACE _trace
#endif

//To make SUNC happy
extern "C" {
	static pthread_key_t runningThreadKey;
	static pthread_once_t runningThreadKey_once = PTHREAD_ONCE_INIT;

	static void make_runningThreadKey()
	{
	   pthread_key_create(&runningThreadKey, NULL);
	}
}


namespace Falcon {
namespace Sys {


//======================================================
// Mutex
//

// for now we ignore spincount
void MutexProvider::init( Mutex *mtx, long )
{
   pthread_mutexattr_t attr;
   pthread_mutexattr_init( &attr );

   pthread_mutex_t *pmutex = (pthread_mutex_t *) memAlloc( sizeof( pthread_mutex_t ) );
   pthread_mutex_init( pmutex, &attr );
   mtx->m_sysData = pmutex;

   pthread_mutexattr_destroy( &attr );
}


void MutexProvider::destroy( Mutex *mtx )
{
   pthread_mutex_t *pmutex = (pthread_mutex_t *) mtx->m_sysData;
   pthread_mutex_destroy( pmutex );
}

bool MutexProvider::trylock( Mutex *mtx )
{
   pthread_mutex_t *pmutex = (pthread_mutex_t *) mtx->m_sysData;
   return pthread_mutex_trylock( pmutex ) != EBUSY;
}

void MutexProvider::lock( Mutex *mtx )
{
   pthread_mutex_t *pmutex = (pthread_mutex_t *) mtx->m_sysData;
   pthread_mutex_lock( pmutex );
}

void MutexProvider::unlock( Mutex *mtx )
{
   pthread_mutex_t *pmutex = (pthread_mutex_t *) mtx->m_sysData;
   pthread_mutex_unlock( pmutex );
}


//======================================================
// Waitable
//

void WaitableProvider::init( Waitable *wo )
{
   wo->m_sysData = new POSIX_WAITABLE(wo);
}

void WaitableProvider::destroy( Waitable *wo )
{
   delete ( POSIX_WAITABLE *) wo->m_sysData;
}


void WaitableProvider::signal( Waitable *wo )
{
   broadcast( wo );
/*
   POSIX_WAITABLE *pwo = (POSIX_WAITABLE *) wo->m_sysData;
   POSIX_TH *pt = 0;
   pthread_mutex_lock( &pwo->m_mtx );
   if ( ! pwo->m_waiting.empty() )
   {
      pt = (POSIX_TH *) pwo->m_waiting.front();
      pwo->m_waiting.popFront();
      pthread_mutex_unlock( &pwo->m_mtx );

      pthread_mutex_lock( &pt->m_mtx );
      if ( --pt->m_refCount == 0 )
      {
         pthread_mutex_unlock( &pt->m_mtx );
         delete pt;
      }
      else {
         pt->m_bSignaled = true;
         pthread_cond_signal( &pt->m_condSignaled );
         pthread_mutex_unlock( &pt->m_mtx );
      }
   }
   else
      pthread_mutex_unlock( &pwo->m_mtx );
*/
}


void WaitableProvider::broadcast( Waitable *wo )
{
   TRACE( "Broadcasting %p\n", wo );

   POSIX_WAITABLE *pwo = (POSIX_WAITABLE *) wo->m_sysData;
   POSIX_TH *pt = 0;
   //pwo->m_waitable->m_mtx.lock();

   while( ! pwo->m_waiting.empty() )
   {
      pt = (POSIX_TH *) pwo->m_waiting.front();
      pwo->m_waiting.popFront();
      //pwo->m_waitable->m_mtx.unlock();
      TRACE( "%p: Notifying thread %p\n", wo, pt );

      pthread_mutex_lock( &pt->m_mtx );
      if ( --pt->m_refCount == 0 )
      {
         pthread_mutex_unlock( &pt->m_mtx );
         delete pt;
      }
      else {
         pt->m_bSignaled = true;
         pthread_cond_signal( &pt->m_condSignaled );
         pthread_mutex_unlock( &pt->m_mtx );
      }

      //pwo->m_waitable->m_mtx.lock();
   }

   //pwo->m_waitable->m_mtx.unlock();
   TRACE( "%p: End broadcast\n", wo );
}


void WaitableProvider::lock( Waitable *wo )
{
   wo->m_mtx.lock();
}


bool WaitableProvider::acquireInternal( Waitable *wo )
{
   return wo->acquireInternal();
}


void WaitableProvider::unlock( Waitable *wo )
{
   wo->m_mtx.unlock();
}


//==============================================
//==============================================

POSIX_WAITABLE::POSIX_WAITABLE( Waitable *wo )
{
   m_waitable = wo;
   //pthread_mutex_init( &m_mtx, NULL );
}

POSIX_WAITABLE::~POSIX_WAITABLE()
{
   //pthread_mutex_destroy( &m_mtx );
}

bool POSIX_WAITABLE::waitOnThis( POSIX_TH *th )
{
   TRACE( "waitOnThis %p\n", th );

   WaitableProvider::lock( m_waitable );

   TRACE( "%p: Trying to acquire\n", th );
   if( WaitableProvider::acquireInternal( m_waitable ) )
   {
      WaitableProvider::unlock( m_waitable );
      TRACE( "%p: Acquired\n", th );
      return true;
   }
   TRACE( "%p: Acquired failed, proceeding\n", th );

   // check if we're still in the waiting list.
   ListElement *le = m_waiting.begin();
   while( le != 0 )
   {
      if ( th == le->data() )
      {
         WaitableProvider::unlock( m_waitable );
         TRACE( "%p: Already waiting\n", th );
         return false;
      }

      le = le->next();
   }

   // we're not; add us
   TRACE( "%p: Goiong to wait\n", th );
   // this is just to ensure correct visibility
   pthread_mutex_lock( &th->m_mtx );
   th->m_refCount++;
   pthread_mutex_unlock( &th->m_mtx );
   m_waiting.pushBack( th );
   WaitableProvider::unlock( m_waitable );

   return false;
}

void POSIX_WAITABLE::cancelWait( POSIX_TH *th )
{
   WaitableProvider::lock( m_waitable );
   ListElement *le = m_waiting.begin();
   while( le != 0 )
   {
      if ( th == le->data() )
      {
         m_waiting.erase( le );
         WaitableProvider::unlock( m_waitable );

         pthread_mutex_lock( &th->m_mtx );
         if ( --th->m_refCount == 0 )
         {
            pthread_mutex_unlock( &th->m_mtx );
            delete th;
         }
         else
            pthread_mutex_unlock( &th->m_mtx );

         return;
      }

      le = le->next();
   }
   WaitableProvider::unlock( m_waitable );
}

//======================================================
// Thread
//

POSIX_TH::POSIX_TH()
{
   pthread_cond_init( &m_condSignaled, NULL );
   pthread_mutex_init( &m_mtx, NULL );
   m_refCount = 1;
   m_bSignaled = false;
   m_bInterrupted = false;
   lastError = 0;
}

POSIX_TH::~POSIX_TH()
{
   pthread_cond_destroy( &m_condSignaled );
   pthread_mutex_destroy( &m_mtx );
}


extern "C" {
static void *s_threadRunner( void *data )
{
   Thread *runner = (Thread *) data;

   // save the runner as specific for this thread
   pthread_setspecific( runningThreadKey, runner );

   void *res = runner->run();
   runner->terminated();
   runner->decref();
   return res;
}
}

void ThreadProvider::initSys()
{
   // Nothing
}

void ThreadProvider::init( Thread *runner )
{
   // This is a good moment to create the thread key.
   pthread_once(&runningThreadKey_once, make_runningThreadKey );

   POSIX_TH *data = new POSIX_TH;
   runner->m_sysData = data;
}

void ThreadProvider::configure( Thread *runner, SystemData &sdt )
{
   // Nothing
}


Thread *ThreadProvider::getRunningThread()
{
   // This is a good moment to create the thread key.
   pthread_once( &runningThreadKey_once, make_runningThreadKey );
   return (Thread *) pthread_getspecific( runningThreadKey );
}

void ThreadProvider::setRunningThread( Thread *th )
{
   // This is a good moment to create the thread key.
   pthread_once( &runningThreadKey_once, make_runningThreadKey );
   pthread_setspecific( runningThreadKey, th );
   if ( th != 0 )
   {
      th->m_mtx.lock();
      th->m_bStarted = true;
      ((POSIX_TH *)th->m_sysData)->thID = pthread_self();
      th->m_mtx.unlock();
   }
}


bool ThreadProvider::start( Thread *runner, const ThreadParams &params )
{
   POSIX_TH *pth = (POSIX_TH *) runner->m_sysData;
   pthread_attr_t attr;

   pthread_attr_init( &attr );
   if( params.stackSize() != 0 )
   {
      if( (pth->lastError =  pthread_attr_setstacksize( &attr, params.stackSize() ) ) != 0 )
      {
         pthread_attr_destroy( &attr );
         return false;
      }
   }

   if ( params.detached() )
   {
      if( (pth->lastError =  pthread_attr_setdetachstate( &attr, params.detached() ? 1:0 ) ) != 0 )
      {
         pthread_attr_destroy( &attr );
         return false;
      }
   }

   // time to increment the reference count of our thread that is going to run
   runner->incref();
   if ( (pth->lastError = pthread_create( &pth->thID, &attr, s_threadRunner, runner ) ) != 0 )
   {
      pthread_attr_destroy( &attr );
      runner->decref();
      return false;
   }

   if ( params.detached() )
   {
      // this detach will have effect only on our internal status; system has already detached us.
      runner->ThreadStatus::detach();
   }

   pthread_attr_destroy( &attr );
   return true;
}


bool ThreadProvider::stop( Thread *runner )
{
   POSIX_TH *pth = (POSIX_TH *) runner->m_sysData;
   return pthread_cancel( pth->thID ) == 0;
}

void ThreadProvider::interruptWaits( Thread *runner )
{
   POSIX_TH *pth = (POSIX_TH *) runner->m_sysData;
   pthread_mutex_lock( &pth->m_mtx );
   if ( ! pth->m_bInterrupted )
   {
      pth->m_bInterrupted = true;
      pth->m_bSignaled = true;;
      pthread_cond_signal( &pth->m_condSignaled );
   }
   pthread_mutex_unlock( &pth->m_mtx );
}

/*
bool ThreadProvider::join( Thread *runner, void *&value )
{
   POSIX_TH *pth = (POSIX_TH *) runner->m_sysData;
   pth->lastError = pthread_join( pth->thID, &value );
   return pth->lastError != 0;
}
*/

bool ThreadProvider::detach( Thread *runner )
{
   POSIX_TH *pth = (POSIX_TH *) runner->m_sysData;
   pth->lastError = pthread_detach( pth->thID );
   return pth->lastError != 0;
}


void ThreadProvider::destroy( Thread *runner )
{
   POSIX_TH *data = (POSIX_TH *) runner->m_sysData;
   pthread_mutex_lock( &data->m_mtx );
   if ( --data->m_refCount == 0 )
   {
      pthread_mutex_unlock( &data->m_mtx );
      delete data;
   }
   else
      pthread_mutex_unlock( &data->m_mtx );
}


uint64 ThreadProvider::getID( const Thread *runner )
{
   POSIX_TH *pth = (POSIX_TH *) runner->m_sysData;
   return (uint64) pth->thID;
}


uint64 ThreadProvider::getCurrentID()
{
   return (uint64) pthread_self();
}


bool ThreadProvider::equal( const Thread *th1, const Thread *th2 )
{
   const POSIX_TH *pth1 = (POSIX_TH *) th1->m_sysData;
   const POSIX_TH *pth2 = (POSIX_TH *) th2->m_sysData;

   return pthread_equal( pth1->thID, pth2->thID ) == 0;
}


bool ThreadProvider::isCurrentThread( const Thread *runner )
{
   const POSIX_TH *pth = (POSIX_TH *) runner->m_sysData;

   return pthread_equal( pth->thID, pthread_self() ) == 0;
}


int ThreadProvider::waitForObjects( const Thread *runner, int32 count, Waitable **objects, int64 time )
{
   struct timespec ts;

   POSIX_TH *data = (POSIX_TH *) runner->m_sysData;

   // first, let's see if we have some data ready
   // in case of no wait, just check for availability

   TRACE( "Entering wait %p\n", data );

   for ( int32 i = 0; i < count; i++ )
   {
      // try-acquire semantic.
      if ( objects[i]->acquire() )
      {
         TRACE( "%p: Acquired item %d\n", data, i );
         return i;
      }
   }

   // noway.
   if ( time == 0 ) {
      TRACE( "%p: Forfaiting (try-aqcuire fail)\n", data );
      return -1;
   }

   // else, if we have to wait for sometime, get the time now.
   // ... in this case, prepare for absolute time wait.
   if ( time > 0 )
   {
      struct timeval    tp;
      gettimeofday(&tp, NULL);
      ts.tv_sec  = tp.tv_sec;
      ts.tv_nsec = tp.tv_usec * 1000;

      ts.tv_sec += time/ 1000000;
      ts.tv_nsec += (time%1000000)*1000;
      if( ts.tv_nsec > 1000000000 )
      {
         ts.tv_nsec -= 1000000000;
         ts.tv_sec ++;
      }
   }

   // acquire with notify-back semanitic.
   data->m_bSignaled = false;

   // now wait till we can acquire something.
   int acquired = -1;
   bool bComplete = false;

   TRACE( "%p: Starting notify-back loop\n", data );

   while( true )
   {
      TRACE( "%p: Notify-back loop begin\n", data );
      // try again to acquire everything.
      for ( int32 i = 0; i < count; i++ )
      {
         POSIX_WAITABLE *pwo = (POSIX_WAITABLE *) objects[i]->m_sysData;

         // try-acquire semantic.
         if ( pwo->waitOnThis( data ) )
         {
            TRACE( "%p: Success on waitOnThis %d\n", data, i );
            acquired = i;
            bComplete = true;
            break;
         }
         TRACE( "%p: Wating on %d\n", data, i );
      }

      if ( bComplete )
         break;

      //wait to be signaled
      pthread_mutex_lock( &data->m_mtx );
      TRACE( "%p: Starting notify-back wait\n", data );
      while( ! data->m_bSignaled )
      {
         TRACE( "%p: Notify-back loop\n", data );
         if ( time > 0 )
         {
            if ( pthread_cond_timedwait( &data->m_condSignaled, &data->m_mtx, &ts ) == ETIMEDOUT )
            {
               // we didn't make it on time. We must stop waiting and return.
               TRACE( "%p: Timeout on notify-back\n", data );
               bComplete = true;
               break;
            }
         }
         else{
            pthread_cond_wait( &data->m_condSignaled, &data->m_mtx );
         }
      }
      data->m_bSignaled = false; // reset for the next loop, if we have one.

      // if we have been interrupted; we should simply return -2 and go away.
      if ( data->m_bInterrupted )
      {
         TRACE( "%p: Thread wait interrupted\n", data );
         data->m_bInterrupted = false; // reset
         pthread_mutex_unlock( &data->m_mtx );
         acquired = -2;
         break;
      }

      pthread_mutex_unlock( &data->m_mtx );

      if ( bComplete )
         break;
   }

   TRACE( "%p: Exiting loop with acquired %d\n", data, acquired );
   // anyhow, unwait everything
   if( count > 1 && acquired <= 0 )
   {
      // but only if we have more than one thing to wait on, or if we failed to acquire
      // otherwise we know we've been removed from the waiting list
      for ( int32 i = 0; i < count; i++ )
      {
         TRACE( "%p: Canceling wait from object %d\n", data, objects );
         POSIX_WAITABLE *pwo = (POSIX_WAITABLE *) objects[i]->m_sysData;
         pwo->cancelWait( data );
      }
   }
   return acquired;
}


}
}

/* end of systhread_posix.cpp */
