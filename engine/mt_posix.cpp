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

#if defined BSD || defined __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <falcon/mt.h>
#include <falcon/atomic.h>

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



void Event::set()
{
   #ifdef NDEBUG
   pthread_mutex_lock( &m_mtx );
   m_bIsSet = true;
   pthread_mutex_unlock( &m_mtx );
   pthread_cond_broadcast( &m_cv );
   #else
   int result = pthread_mutex_lock( &m_mtx );
   fassert( result == 0 );
   m_bIsSet = true;
   result = pthread_cond_broadcast( &m_cv );
   fassert( result == 0 );
   result = pthread_mutex_unlock( &m_mtx );
   fassert( result == 0 );
   #endif
}


bool Event::wait( int32 to )
{
   if( to == 0 )
   {
      pthread_mutex_lock( &m_mtx );
   }
   else if( to < 0 )
   {
      pthread_mutex_lock( &m_mtx );
      while( ! m_bIsSet ) {
         pthread_cond_wait( &m_cv, &m_mtx );
      }
   }
   else {
      struct timeval tv;
      struct timespec ts;
      gettimeofday( &tv, NULL );

      ts.tv_nsec = (tv.tv_usec + ((to%1000)*1000))*1000;
      ts.tv_sec = tv.tv_sec + (to/1000);
      if( ts.tv_nsec >= 1000000000 ) {
         ts.tv_sec++;
         ts.tv_nsec -= 1000000000;
      }
      pthread_mutex_lock( &m_mtx );

      while( ! m_bIsSet ) {
         int res = pthread_cond_timedwait( &m_cv, &m_mtx, &ts );
         if( res == ETIMEDOUT ) {
            break;
         }
      }
   }

   bool result = m_bIsSet;
   if( m_bAutoReset ) {
      m_bIsSet = false;
   }
   pthread_mutex_unlock( &m_mtx );

   return result;
}

//==================================================================================
// System threads.
//

SysThread::~SysThread()
{
   pthread_mutex_destroy( &m_sysdata->m_mtxT );
   free( m_sysdata );
}

SysThread::SysThread( Runnable* r ):
   m_runnable( r )
{
   m_sysdata = ( struct SYSTH_DATA* ) malloc( sizeof( struct SYSTH_DATA ) );
   m_sysdata->m_bDetached = false;
   m_sysdata->m_bDone = false;
   m_sysdata->m_lastError = 0;
   pthread_mutex_init( &m_sysdata->m_mtxT, NULL );
}

void SysThread::attachToCurrent()
{
   m_sysdata->pth = pthread_self();
}

uint32 SysThread::lastError() const
{
   return (uint32) m_sysdata->m_lastError;
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


//==========================================================
// Interruptible event
//==========================================================

struct int_evt
{
   pthread_cond_t cond;
   pthread_mutex_t mtx;

   bool autoReset;
   bool isSet;
   bool isInterrupted;

};


InterruptibleEvent::InterruptibleEvent( bool bManualReset, bool initState )
{
   struct int_evt* evt = new struct int_evt;
   m_sysdata = evt;
   evt->isInterrupted = false;
   evt->isSet = initState;
   evt->autoReset = !bManualReset;

   pthread_cond_init( &evt->cond, 0);
   pthread_mutex_init( &evt->mtx, 0);
}


InterruptibleEvent::~InterruptibleEvent()
{
   struct int_evt* evt = (struct int_evt*) m_sysdata;
   pthread_cond_destroy(&evt->cond);
   pthread_mutex_destroy(&evt->mtx);

   delete evt;
}


void InterruptibleEvent::set()
{
   struct int_evt* evt = (struct int_evt*) m_sysdata;

   pthread_mutex_lock( &evt->mtx );
   evt->isSet = true;
   pthread_mutex_unlock( &evt->mtx );
   pthread_cond_broadcast( &evt->cond );
}


InterruptibleEvent::wait_result_t InterruptibleEvent::wait( int32 to )
{
   struct int_evt* evt = (struct int_evt*) m_sysdata;
   wait_result_t result;

   if( to == 0 )
   {
      pthread_mutex_lock( &evt->mtx );

   }
   else if( to < 0 )
   {
      pthread_mutex_lock( &evt->mtx );
      while( !(evt->isInterrupted|| evt->isSet) ) {
         pthread_cond_wait( &evt->cond, &evt->mtx );
      }
   }
   else {
      struct timeval tv;
      struct timespec ts;
      gettimeofday( &tv, NULL );

      ts.tv_nsec = (tv.tv_usec + ((to%1000)*1000))*1000;
      ts.tv_sec = tv.tv_sec + (to/1000);
      if( ts.tv_nsec > 1000000000 )
      {
         ts.tv_nsec -= 1000000000;
         ts.tv_sec += 1;
      }
      pthread_mutex_lock( &evt->mtx );

      while( !(evt->isInterrupted|| evt->isSet) ) {
         int res = pthread_cond_timedwait( &evt->cond, &evt->mtx, &ts );
         if( res == ETIMEDOUT ) {
            break;
         }
      }
   }

   if( evt->isInterrupted ) {
      result = wait_interrupted;
   }
   else if( evt->isSet ) {
      result = wait_success;
      if( evt->autoReset ) {
         evt->isSet = false;
      }
   }
   else {
      result = wait_timedout;
   }
   pthread_mutex_unlock( &evt->mtx );

   return result;
}


void InterruptibleEvent::interrupt()
{
   struct int_evt* evt = (struct int_evt*) m_sysdata;

   pthread_mutex_lock( &evt->mtx );
   evt->isInterrupted = true;
   pthread_mutex_unlock( &evt->mtx );
   pthread_cond_broadcast( &evt->cond );
}

void InterruptibleEvent::reset()
{
   struct int_evt* evt = (struct int_evt*) m_sysdata;

   pthread_mutex_lock( &evt->mtx );
   evt->isInterrupted = false;
   evt->isSet = false;
   pthread_mutex_unlock( &evt->mtx );
}

}

/* end of mt_posix.cpp */
