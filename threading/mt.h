/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: mt.h

   Multithreading abstraction layer
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 08 Apr 2008 20:10:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Multithreading abstraction layer.

   Mutexes, condition variables and threads. Very basic things.
*/

#ifndef FLC_MT_H
#define FLC_MT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>
#include <falcon/genericlist.h>
#include <falcon/error.h>

#include <systhread.h>

namespace Falcon {

class Item;

namespace Sys {

typedef int (*threadRoutine)( void *param );

class Mutex: public BaseAlloc
{
   friend class MutexProvider;
   void *m_sysData;

public:
   Mutex( long spincount = 0 ) { MutexProvider::init( this, spincount); }
   ~Mutex() { MutexProvider::destroy( this ); }

   void lock() { MutexProvider::lock( this ); }
   void unlock() { MutexProvider::unlock( this ); }
   bool trylock() { return MutexProvider::trylock( this ); }
};

/** Waiter class
   This class waits for something to happen...
*/
class Waitable: public BaseAlloc
{
   friend class WaitableProvider;
   friend class ThreadProvider;
   void *m_sysData;


protected:
   Waitable():
      m_refCount( 1 )
   {
      WaitableProvider::init( this );
   }

   virtual ~Waitable() { WaitableProvider::destroy( this ); }

   /** Signal to the first interested thread that we may be acquired. */
   void signal() { WaitableProvider::signal( this ); }

   /** Signal to all the interested thread that we may be acquired. */
   void broadcast() { WaitableProvider::broadcast( this ); }

   /** Useful for subclasses */
   mutable Mutex m_mtx;
   mutable int m_refCount;

   virtual bool acquireInternal()=0;

public:

   virtual bool acquire() = 0;
   virtual void release() = 0;

   void incref();
   void decref();
};


class Grant: public Waitable
{
   int m_count;

protected:
   virtual bool acquireInternal();

public:
   Grant( int count = 1 );
   virtual ~Grant();

   virtual bool acquire();
   virtual void release();
};

class Barrier: public Waitable
{
   bool m_bOpen;

protected:
   virtual bool acquireInternal();

public:
   Barrier( bool bOpen = false );
   virtual ~Barrier();

   virtual bool acquire();
   virtual void release();

   void open();
   void close();
};


/** Windows like events. */
class Event: public Waitable
{
   bool m_bSignaled;
   bool m_bAutoReset;
   bool m_bHeld;

protected:
   virtual bool acquireInternal();

public:
   Event( bool bAutoReset = true );
   virtual ~Event();
   virtual bool acquire();
   virtual void release();

   virtual void set();
   virtual void reset();
};

/** Synchronized queue. */
class SyncQueue: public Waitable
{
   List m_items;
   bool m_bHeld;

protected:
   virtual bool acquireInternal();

public:
   SyncQueue();
   virtual ~SyncQueue();

   virtual bool acquire();
   virtual void release();

   virtual void pushFront( void *data );
   virtual void pushBack( void *data );
   virtual bool popFront( void *&data );
   virtual bool popBack( void *&data );
   virtual bool empty() const;
   virtual uint32 size() const;
};

/** Counter (semaphore). */
class SyncCounter: public Waitable
{
   int m_count;

protected:
   virtual bool acquireInternal();

public:
   SyncCounter( int init = 0 );
   virtual ~SyncCounter();

   virtual bool acquire();
   virtual void release();
   void post( int count = 1 );
};


/** Thread status. */
class ThreadStatus: public Waitable
{

protected:
   bool m_bTerminated;
   bool m_bDetached;
   bool m_bStarted;
   int m_acquiredCount;

protected:
   virtual bool acquireInternal();

public:
   ThreadStatus();
   virtual ~ThreadStatus();

   virtual bool acquire();
   virtual void release();

   bool isTerminated() const;
   bool isDetached() const;

   /** Turns the status into started if not started nor detached. */
   bool startable();

   bool detach();
   /** Turn the status in terminated, removing the started status. */
   bool terminated();
};

//*****************************************
// Thread initialization parameters
//

class ThreadParams: public BaseAlloc
{
   uint32 m_stackSize;
   bool m_bDetached;

public:
   ThreadParams():
      m_stackSize(0),
      m_bDetached( false )
   {}

   ThreadParams &stackSize( uint32 size ) { m_stackSize = size; return *this; }
   ThreadParams &detached( bool setting ) { m_bDetached = setting; return *this; }

   uint32 stackSize() const { return m_stackSize; }
   bool detached() const { return m_bDetached; }
};


/** Thread base class.
   Derive your thread class from this one and overload the run() method to
   have Falcon engine multiplatform threading support.

   All the functionalities of this class are provided by ThreadProvider;
   the vast majority of the calls in this class are a direct inline
   shortcut to the ThreadProvider, that will provide system-dependent threading.
*/
class Thread: public ThreadStatus
{
   friend class ThreadProvider;

protected:
   int m_nRefCount;
   void *m_sysData;

   Thread():
      m_nRefCount( 1 )
   { ThreadProvider::init( this ); }

public:
   virtual ~Thread();

   bool start( const ThreadParams &params ) { return ThreadProvider::start( this, params ); }
   bool start() { return ThreadProvider::start( this, ThreadParams() ); }
   bool stop() { return ThreadProvider::stop( this ); }
   void interruptWaits() { ThreadProvider::interruptWaits( this ); }

   /** Warning; this is not a virtual method, it's just an override for this class. */
   bool detach() {
      if ( ThreadStatus::detach() )
      {
         ThreadProvider::detach( this );
         return true;
      }
      return false;
   }

   uint64 getID() const { return ThreadProvider::getID( this ); }
   static uint64 getCurrentID() { return ThreadProvider::getCurrentID(); }
   bool equal( const Thread &other ) const { return ThreadProvider::equal( this, &other ); }
   bool isCurrentThread() const { return ThreadProvider::isCurrentThread( this ); }

   /** Wait for objects to be signaled.
      This is relatively similar to the MS-SDK WaitForMultipleObjects() function, but
      it uses posix cancelation semantics and its bound to the thread where the wait
      is performed.
   */
   int waitForObjects( int32 count, Waitable **objects, int64 time=-1 )
   {
      return ThreadProvider::waitForObjects( this, count, objects, time );
   }

   void wakeUp( Waitable *object ) { ThreadProvider::wakeUp( this, object ); }

   void incref();
   void decref();

   virtual void *run()=0;
};




}
}

#endif

/* end of mt.h */
