/*
   FALCON - The Falcon Programming Language.
   FILE: mt_posix.cpp

   Multithreaded extensions - POSIX specific.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 17 Jan 2009 17:06:47 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define _XOPEN_SOURCE_EXTENDED
#include <time.h>

// this for gettimeofday on macosx
#include <sys/time.h>

#include <falcon/mt.h>
#include <falcon/memory.h>

namespace Falcon
{

static Mutex s_cs;

ThreadSpecific::ThreadSpecific( void (*destructor)(void*) )
{
   #ifndef NDEBUG
   int value = pthread_key_create( &m_key, destructor );
   fassert( value == 0 );
   #else
   pthread_key_create( &m_key, destructor );
   #endif
}

/** Performs an atomic thread safe increment. */
int32 atomicInc( volatile int32 &data )
{
   s_cs.lock();
   register int32 res = ++data;
   s_cs.unlock();
   return res;
}

/** Performs an atomic thread safe decrement. */
int32 atomicDec( volatile int32 &data )
{
   s_cs.lock();
   register int32 res = --data;
   s_cs.unlock();
   return res;
}



void Event::set()
{
   #ifdef NDEBUG
   pthread_mutex_lock( &m_mtx );
   m_bIsSet = true;

   if ( m_bAutoReset )
      pthread_cond_signal( &m_cv );
   else
      pthread_cond_broadcast( &m_cv );

   pthread_mutex_unlock( &m_mtx );
   #else
   int result = pthread_mutex_lock( &m_mtx );
   fassert( result == 0 );
   m_bIsSet = true;

   if ( m_bAutoReset )
      result = pthread_cond_signal( &m_cv );
   else
      result = pthread_cond_broadcast( &m_cv );

   fassert( result == 0 );
   result = pthread_mutex_unlock( &m_mtx );
   fassert( result == 0 );
   #endif
}


bool Event::wait( int32 to )
{
   pthread_mutex_lock( &m_mtx );

   // are we lucky?
   if( m_bIsSet )
   {
      if ( m_bAutoReset )
         m_bIsSet = false;
      pthread_mutex_unlock( &m_mtx );
      return true;
   }

   // No? -- then are we unlucky?
   if ( to == 0 )
   {
      pthread_mutex_unlock( &m_mtx );
      return false;
   }

   // neither? -- then let's wait. How much?
   if ( to <  0 )
   {
      do {
         pthread_cond_wait( &m_cv, &m_mtx );

      } while( ! m_bIsSet );
   }
   else
   {
      // release the mutex for a while
      pthread_mutex_unlock( &m_mtx );

      struct timespec ts;
      #if _POSIX_TIMERS > 0
         clock_gettime(CLOCK_REALTIME, &ts);
      #else
          struct timeval tv;
          gettimeofday(&tv, NULL);
          ts.tv_sec = tv.tv_sec;
          ts.tv_nsec = tv.tv_usec*1000;
      #endif

      ts.tv_sec += to/1000;
      ts.tv_nsec += (to%1000) * 1000000;
      if( ts.tv_nsec >= 1000000000 )
      {
         ++ts.tv_sec;
         ts.tv_nsec -= 1000000000;
      }

      pthread_mutex_lock( &m_mtx );
      while( ! m_bIsSet )
      {
         int res;
         if( (res = pthread_cond_timedwait( &m_cv, &m_mtx, &ts )) == ETIMEDOUT )
         {
            // wait failed
            pthread_mutex_unlock( &m_mtx );
            return false;
         }
         // be sure that we haven't got other reasons to fail.
         fassert( res == 0 );
      }
   }

   // here, m_bIsSet is set...
   if ( m_bAutoReset )
      m_bIsSet = false;
   pthread_mutex_unlock( &m_mtx );

   return true;
}

//==================================================================================
// System threads.
//

SysThread::~SysThread()
{
   pthread_mutex_destroy( &m_sysdata->m_mtxT );
   memFree( m_sysdata );
}

SysThread::SysThread( Runnable* r ):
   m_runnable( r )
{
   m_sysdata = ( struct SYSTH_DATA* ) memAlloc( sizeof( struct SYSTH_DATA ) );
   m_sysdata->m_bDetached = false;
   m_sysdata->m_bDone = false;
   m_sysdata->m_lastError = 0;
   pthread_mutex_init( &m_sysdata->m_mtxT, NULL );
}

void SysThread::attachToCurrent()
{
   m_sysdata->pth = pthread_self();
}



void* SysThread::RunAThread( void *data )
{
   SysThread* sth = (SysThread*) data;
   return sth->run();
}

bool SysThread::start( const ThreadParams &params )
{
   pthread_attr_t attr;

   pthread_attr_init( &attr );
   if( params.stackSize() != 0 )
   {
      if( (m_sysdata->m_lastError =  pthread_attr_setstacksize( &attr, params.stackSize() ) ) != 0 )
      {
         pthread_attr_destroy( &attr );
         return false;
      }
   }

   if ( params.detached() )
   {
      if( (m_sysdata->m_lastError =  pthread_attr_setdetachstate( &attr, params.detached() ? 1:0 ) ) != 0 )
      {
         pthread_attr_destroy( &attr );
         return false;
      }
   }

   // time to increment the reference count of our thread that is going to run
   if ( (m_sysdata->m_lastError = pthread_create( &m_sysdata->pth, &attr, &SysThread::RunAThread, this ) ) != 0 )
   {
      pthread_attr_destroy( &attr );
      return false;
   }
   
   if ( params.detached() )
   {
      detach();
   }

   pthread_attr_destroy( &attr );
   return true;
}

void SysThread::disengage()
{
   delete this;
}

void SysThread::detach()
{
   // are we already done?
   pthread_mutex_lock( &m_sysdata->m_mtxT );
   if ( m_sysdata->m_bDone ) 
   {
      pthread_mutex_unlock( &m_sysdata->m_mtxT );
      // disengage system joins and free system data.
      pthread_detach( m_sysdata->pth );
      // free app data.
      delete this;
   }
   else {
      // tell the run function to dispose us when done.
      m_sysdata->m_bDetached = true;
      pthread_mutex_unlock( &m_sysdata->m_mtxT );
   }
}

bool SysThread::join( void* &result )
{
   if ( pthread_join( m_sysdata->pth, &result ) == 0 )
   {
      delete this;
      return true;
   }
   return false;
}


uint64 SysThread::getID()
{
   return (uint64) m_sysdata->pth;
}

uint64 SysThread::getCurrentID()
{
   return (uint64) pthread_self();
}

bool SysThread::isCurrentThread()
{
   return pthread_equal( m_sysdata->pth, pthread_self() ) != 0;
}

bool SysThread::equal( const SysThread *th1 ) const
{
   return pthread_equal( m_sysdata->pth, th1->m_sysdata->pth ) != 0;
}

void *SysThread::run()
{
   fassert( m_runnable !=  0 );
   void* data = m_runnable->run();
   
   // have we been detached in the meanwhile? -- we must dispose our data now.
   pthread_mutex_lock( &m_sysdata->m_mtxT );
   if( m_sysdata->m_bDetached ) 
   {
      pthread_mutex_unlock( &m_sysdata->m_mtxT );
      delete this;
   }
   else {
      m_sysdata->m_bDone = true;
      pthread_mutex_unlock( &m_sysdata->m_mtxT );
   }
   
   return data;
}

}

/* end of mt_posix.cpp */
