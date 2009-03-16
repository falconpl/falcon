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
#include <mt.h>

namespace Falcon {
namespace Ext{

//======================================================
//

/** This is the class that will run the target thread.
*/
class VMRunnerThread: public ::Falcon::Sys::Thread
{
   VMachine *m_vm;
   bool m_bOwn;
   Item m_threadInstance;
   Item m_method;
   Error *m_lastError;

   ~VMRunnerThread();
public:

   /** Creates the VM thread -- and an empty VM. */
   VMRunnerThread();

   /** Adopt a running VM in a threading shell. */
   VMRunnerThread( VMachine *vm );


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

   const VMachine &vm() const { return *m_vm; }
   VMachine &vm() { return *m_vm; }
   virtual void *run();

   bool hadError() const { return m_lastError != 0; }
   Error* exitError() const { return m_lastError; }
};

//======================================================
//

class WaitableCarrier: public FalconData
{
protected:
   ::Falcon::Sys::Waitable *m_wto;

public:
   WaitableCarrier( ::Falcon::Sys::Waitable *t );
   WaitableCarrier( const WaitableCarrier &other );
   virtual ~WaitableCarrier();
   virtual FalconData *clone() const;
   virtual void gcMark( ::Falcon::MemPool* ) {}

   ::Falcon::Sys::Waitable *waitable() const { return m_wto; }
};

//======================================================
//

class ThreadCarrier: public WaitableCarrier
{

public:
   ThreadCarrier( VMRunnerThread *t );
   ThreadCarrier( const ThreadCarrier &other );

   virtual ~ThreadCarrier();
   virtual FalconData *clone() const;

   VMRunnerThread *thread() const { return static_cast<VMRunnerThread *>(m_wto); }
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


}
}

#endif

/* end of threading_mod.h */
