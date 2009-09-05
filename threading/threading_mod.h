/*
   FALCON - The Falcon Programming Language.
   FILE: threading_mod.h

   Threading module binding extensions - internal implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 10 Apr 2008 23:26:43 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Threading module binding extensions - internal implementation.
*/

#ifndef FLC_THREADING_MOD
#define FLC_THREADING_MOD

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include <falcon/item.h>
#include <falcon/error.h>
#include <falcon/vm.h>
#include <falcon/mt.h>
#include <waitable.h>

namespace Falcon {
namespace Ext{

class ThreadImpl: public Runnable, public BaseAlloc
{
protected:
   int m_nRefCount;
   SysThread* m_sth;
   ThreadStatus m_thstatus;
   
   void* m_sysData;
   
   VMachine *m_vm;
   Error *m_lastError;
   Item m_threadInstance;
   Item m_method;
   
   int m_id;
   String m_name;
   
public:
   ThreadImpl();
   ThreadImpl( const String& name );
   virtual ~ThreadImpl();
   
   /** Adopt a running VM in a threading shell. */
   ThreadImpl( VMachine *vm );
   
   bool startable() { return m_thstatus.startable(); }
   bool isTerminated() const { return m_thstatus.isTerminated(); }
   bool isDetached() const { return m_thstatus.isDetached(); }
   bool acquire() { return m_thstatus.acquire(); }
   void release() { m_thstatus.release(); }
   
   bool start( const ThreadParams &params );
   bool start() { return start( ThreadParams() ); }
   void interruptWaits()
   {
      return WaitableProvider::interruptWaits( this );
   }
   
   /** Wait for objects to be signaled.
      This is relatively similar to the MS-SDK WaitForMultipleObjects() function, but
      it uses posix cancelation semantics and its bound to the thread where the wait
      is performed.
   */
   int waitForObjects( int32 count, Waitable **objects, int64 time=-1 )
   { 
      return WaitableProvider::waitForObjects( this, count, objects, time ); 
   }

   bool join();
   bool detach();
   void disengage() { m_sth->disengage(); m_sth = 0; }
   int getID() const { return m_id; }
   /** Return the name set for this thread. */
   const String& name() const { return m_name; }
   void name( const String &name ) { m_name = name; }
   
   uint64 getSystemID() const { return m_sth == 0 ? 0 : m_sth->getID(); }
   bool equal( const ThreadImpl &other ) const { return m_sth->equal( other.m_sth ); }
   bool isCurrentThread() const { return m_sth->isCurrentThread(); }

   void incref();
   void decref();


   /** Prepare the running instance.

      This method is meant to be called after the calling thread has acquired
      the right to start the thread through a successful call to readyToStart().

      At that point, is safe to query this object for the VM and link the needed
      modules in the VM.

      As start() will call the run() method of the VM,
      before launching the machine, this method will claim a garbage lock to
      the Thread instance in the target VM representing this thread. That instance
      holds a thread carrire which holds a reference
      to the ThreadData, which is the final owner of our VM.

      \param instance the item representing the instance of this thread in the VM held by this object.
      \param method the item that will be run in th target VM (self.run).
   */
   void prepareThreadInstance( const Item &instance, const Item &method );

   virtual void *run();

   const VMachine &vm() const { return *m_vm; }
   VMachine &vm() { return *m_vm; }
   

   bool hadError() const { return m_lastError != 0; }
   Error* exitError() const { return m_lastError; }
   void* sysData() const { return m_sysData; }
   ThreadStatus &status() { return m_thstatus; }
};


//======================================================
//

class WaitableCarrier: public FalconData
{
protected:
   Waitable *m_wto;

public:
   WaitableCarrier( Waitable *t );
   WaitableCarrier( const WaitableCarrier &other );
   virtual ~WaitableCarrier();
   virtual FalconData *clone() const;
   virtual void gcMark( ::Falcon::uint32  ) {}

   Waitable *waitable() const { return m_wto; }
};

//======================================================
//

class ThreadCarrier: public FalconData
{
protected:
   ThreadImpl* m_thi;
   
public:
   ThreadCarrier( ThreadImpl *t );
   ThreadCarrier( const ThreadCarrier &other );

   virtual ~ThreadCarrier();
   virtual FalconData *clone() const;
   virtual void gcMark( ::Falcon::uint32  ) {}

   ThreadImpl *thread() const { return m_thi; }
   const VMachine &vm() const { return thread()->vm(); }
   VMachine &vm() { return thread()->vm(); }
};


//======================================================
//

class ThreadError: public ::Falcon::Error
{
public:
   ThreadError():
      Error( "ThreadError" )
   {}

   ThreadError( const ErrorParam &params  ):
      Error( "ThreadError", params )
      {}
};

//======================================================
//

class JoinError: public ::Falcon::Error
{
public:
   JoinError():
      Error( "JoinError" )
   {}

   JoinError( const ErrorParam &params  ):
      Error( "JoinError", params )
      {}
};

//======================================================
// Service functions:
// 

extern void setRunningThread( ThreadImpl* th );
extern ThreadImpl* getRunningThread();

}
}

#endif

/* end of threading_mod.h */
