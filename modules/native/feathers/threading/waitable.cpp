/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: mt.cpp

   Multithreading abstraction layer - common implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 14 Apr 2008 19:25:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Multithreading abstraction layer - common implementation.

   Mutexes, condition variables and threads. Very basic things.
*/

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>
#include <falcon/genericlist.h>
#include <falcon/memory.h>
#include <waitable.h>

#include <systhread.h>

namespace Falcon {
namespace Ext {

void Waitable::incref()
{
   m_mtx.lock();
   m_refCount++;
   m_mtx.unlock();
}

void Waitable::decref()
{
   m_mtx.lock();
   bool bDel = (--m_refCount == 0 );
   m_mtx.unlock();

   if ( bDel )
      delete this;
}

//=====================================
// Grant
//

Grant::Grant( int count ):
   m_count( count )
{}

Grant::~Grant()
{
}

bool Grant::acquire()
{
   m_mtx.lock();
   if( m_count > 0 )
   {
      m_count--;
      m_mtx.unlock();
      return true;
   }
   m_mtx.unlock();

   return false;
}


bool Grant::acquireInternal()
{
   if( m_count > 0 )
   {
      m_count--;
      return true;
   }

   return false;
}


void Grant::release()
{
   m_mtx.lock();
   m_count++;
   if ( m_count == 1 )
   {
      signal();
   }
   else if ( m_count > 1 )
   {
      broadcast();
   }

   m_mtx.unlock();
}


//=====================================
// Barrier
//

Barrier::Barrier( bool bOpen ):
   m_bOpen( bOpen )
{}

Barrier::~Barrier()
{
}

bool Barrier::acquire()
{
   m_mtx.lock();
   bool bStatus = m_bOpen;
   m_mtx.unlock();
   return bStatus;
}


bool Barrier::acquireInternal()
{
   return m_bOpen;
}

void Barrier::release()
{
   // no-op
}

void Barrier::open()
{
   m_mtx.lock();
   m_bOpen = true;
   broadcast();
   m_mtx.unlock();
}

void Barrier::close()
{
   m_mtx.lock();
   m_bOpen = false;
   m_mtx.unlock();
}

//=====================================
// Event
//

Event::Event( bool bAutoReset ):
   m_bSignaled( false ),
   m_bAutoReset( bAutoReset ),
   m_bHeld( false )
{
}

Event::~Event()
{
}


bool Event::acquire()
{
   bool bSuccess;

   m_mtx.lock();
   // a non-auto reset event is currently held?
   if ( m_bHeld )
   {
      bSuccess = false;
   }
   else {
      bSuccess = m_bSignaled;
      if ( m_bAutoReset )
         m_bSignaled = false;
      else
         m_bHeld = true;
   }
   m_mtx.unlock();

   return bSuccess;
}


bool Event::acquireInternal()
{
   bool bSuccess;

   // a non-auto reset event is currently held?
   if ( m_bHeld )
   {
      bSuccess = false;
   }
   else {
      bSuccess = m_bSignaled;
      if ( m_bAutoReset )
         m_bSignaled = false;
      else
         m_bHeld = true;
   }

   return bSuccess;
}

void Event::release()
{
   m_mtx.lock();
   m_bHeld = false;
   // wake another thread?
   bool bSignal = m_bSignaled;
   if( bSignal )
      signal();
   m_mtx.unlock();
}


void Event::set()
{
   bool bSignal;
   m_mtx.lock();
   bSignal = ! m_bSignaled;
   m_bSignaled = true;
   if ( bSignal )
   {
      signal();
   }

   m_mtx.unlock();
}

void Event::reset()
{
   m_mtx.lock();
   m_bSignaled = false;
   m_mtx.unlock();
}

//=====================================
// Counter
//
SyncCounter::SyncCounter( int iCount ):
   m_count( iCount >= 0 ? iCount : 0 )
{
}

SyncCounter::~SyncCounter()
{
}

bool SyncCounter::acquire()
{
   bool bSuccess;

   m_mtx.lock();
   // a non-auto reset event is currently held?
   if ( m_count == 0 )
   {
      bSuccess = false;
   }
   else {
      bSuccess = true;
      --m_count;
   }
   m_mtx.unlock();

   return bSuccess;
}


bool SyncCounter::acquireInternal()
{
   // a non-auto reset event is currently held?
   if ( m_count == 0 )
   {
      return false;
   }
   --m_count;
   return true;
}

void SyncCounter::release()
{
   m_mtx.lock();
   ++m_count;
   signal();
   m_mtx.unlock();
}

void SyncCounter::post( int count )
{
   if ( count <= 0 )
   {
      return;
   }

   m_mtx.lock();
   m_count+=count;
   if ( m_count > 1 )
      broadcast();
   else
      signal();
   m_mtx.unlock();

}

//=====================================
// Thread statatus
//
ThreadStatus::ThreadStatus():
   m_bTerminated( false ),
   m_bDetached( false ),
   m_bStarted( false ),
   m_acquiredCount( 0 )
{}

ThreadStatus::~ThreadStatus()
{}

bool ThreadStatus::acquire()
{
   m_mtx.lock();
   bool bStatus;
   if ( m_bTerminated || m_bDetached )
   {
      bStatus = true;
      m_acquiredCount++;
   }
   else {
      bStatus = false;
   }
   m_mtx.unlock();
   return bStatus;
}

bool ThreadStatus::acquireInternal()
{
   if ( m_bTerminated || m_bDetached )
   {
      m_acquiredCount++;
      return true;
   }
   return false;
}


void ThreadStatus::release()
{
   m_mtx.lock();
   --m_acquiredCount;
   m_mtx.unlock();
}


bool ThreadStatus::isTerminated() const
{
   m_mtx.lock();
   bool bStatus = m_bTerminated;
   m_mtx.unlock();
   return bStatus;
}


bool ThreadStatus::isDetached() const
{
   m_mtx.lock();
   bool bStatus = m_bDetached;
   m_mtx.unlock();
   return bStatus;
}

bool ThreadStatus::detach()
{
   bool bSignal;

   m_mtx.lock();
   if( ! m_bDetached && ! m_bTerminated )
   {
      m_bDetached = true;
      bSignal = true;
      broadcast();

   }
   else
      bSignal = false;
   m_mtx.unlock();

   return bSignal;
}

bool ThreadStatus::startable()
{
   bool bStatus;

   m_mtx.lock();
   if( (! m_bDetached) && (! m_bStarted) && m_acquiredCount == 0 )
   {
      m_bTerminated = false;
      m_bStarted = true;
      bStatus = true;
   }
   else
      bStatus = false;

   m_mtx.unlock();

   return bStatus;
}

bool ThreadStatus::terminated()
{
   bool bSignal;

   m_mtx.lock();
   if( ! m_bDetached && ! m_bTerminated )
   {
      m_bTerminated = true;
      m_bStarted = false;
      bSignal = true;
      broadcast();
   }
   else
      bSignal = false;

   m_mtx.unlock();

   return bSignal;
}


//=========================================================
// SyncQueue
//

SyncQueue::SyncQueue():
   m_bHeld( false )
{}


SyncQueue::~SyncQueue()
{
   m_mtx.lock();
   m_bHeld = true;
   // empty the queue
   ListElement *e = m_items.begin();
   while( e != 0 )
   {
      free( const_cast< void *>(e->data()) );
      e = e->next();
   }
   m_mtx.unlock();
}


bool SyncQueue::acquire()
{
   // try to acquire
   m_mtx.lock();
   if ( m_bHeld || m_items.empty() )
   {
      m_mtx.unlock();
      return false;
   }
   // ok, we acquired.
   m_bHeld = true;
   m_mtx.unlock();

   return true;
}


bool SyncQueue::acquireInternal()
{
   // try to acquire
   if ( m_bHeld || m_items.empty() )
   {
      return false;
   }
   // ok, we acquired.
   m_bHeld = true;
   return true;
}


void SyncQueue::release()
{
   bool bSignal;
   m_mtx.lock();
   // release
   m_bHeld = false;
   // have we still something to say?
   bSignal = ! m_items.empty();
   if ( bSignal )
      signal();
   m_mtx.unlock();
}

void SyncQueue::pushFront( void *data )
{
   bool bSignal;
   m_mtx.lock();
   // was the queue empty?
   bSignal = m_items.empty();
   m_items.pushFront( data );
   if ( bSignal )
      signal();
   m_mtx.unlock();
}


void SyncQueue::pushBack( void *data )
{
   bool bSignal;
   m_mtx.lock();
   // was the queue empty?
   bSignal = m_items.empty();
   m_items.pushFront( data );
   if ( bSignal )
      signal();
   m_mtx.unlock();
}


bool SyncQueue::popFront( void *&data )
{
   bool bSuccess;
   m_mtx.lock();
   if ( m_items.empty() )
   {
      bSuccess = false;
   }
   else
   {
      bSuccess = true;
      data = const_cast< void *>(m_items.front());
      m_items.popFront();
   }
   m_mtx.unlock();

   return bSuccess;
}

bool SyncQueue::popBack( void *&data )
{
   bool bSuccess;
   m_mtx.lock();
   if ( m_items.empty() )
   {
      bSuccess = false;
   }
   else
   {
      bSuccess = true;
      data = const_cast< void *>(m_items.back());
      m_items.popBack();
   }
   m_mtx.unlock();

   return bSuccess;
}


bool SyncQueue::empty() const
{
   m_mtx.lock();
   bool bEmpty = m_items.empty();
   m_mtx.unlock();
   return bEmpty;
}


uint32 SyncQueue::size() const
{
   m_mtx.lock();
   uint32 nSize = m_items.empty();
   m_mtx.unlock();
   return nSize;
}

}
}

/* end of mt.cpp */
