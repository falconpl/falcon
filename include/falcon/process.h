/*
   FALCON - The Falcon Programming Language.
   FILE: process.h

   Falcon virtual machine -- process entity.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 18:51:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PROCESS_H_
#define _FALCON_PROCESS_H_

#include <falcon/setup.h>
#include <falcon/refcounter.h>
#include <falcon/mt.h>
#include <falcon/item.h>

namespace Falcon {

class VMachine;
class VMContext;
class Function;
class Closure;
class Item;
class SynFunc;
class Error;
class GCLock;

/** Process Entity.

 This entity represents an execution process in a virtual machine.

 The process has a VMContext which is the main context, it can be
 waited upon, interrupted asynchronously and checked for a result.

 The constructor is typically created by the virtual machine, which
 sets up a main context for the process and prepares it to run the
 desired code and/or symbol.

 The process can be used to either invoke the execution of a function
 (as i.e. the __main__ function of a module), or executable code stored
 in an arbitrary item, or it can be used to execute PSteps stored
 directly in the associated VM context.

 In the latter case, keep in mind that the Process class itself doesn't
 keep track of the PSteps; the calling program must ensure that the
 PStep instances are alive for the whole duration of the process.

 @note This can be easily ensured by subclassing the Process class
 and storing the required PSteps as process-specific data.
 */
class FALCON_DYN_CLASS Process
{
public:
   Process( VMachine* owner );

   int32 id() const { return m_id; }
   void id( int32 i ) { m_id = i; }

   VMachine* vm() const { return m_vm; }
   VMContext* mainContext() const { return m_context; }

   /** Starts the process execution with the context configured as-is. */
   bool start();
   /** Starts the process invoking the given function. */
   bool start( Function* main, int pcount = 0, Item const* params = 0 );
   /** Starts the process invoking the given closure. */
   bool start( Closure* main, int pcount = 0, Item const* params = 0 );
   /** Starts the process invoking the given item. */
   bool startItem( Item& main, int pcount=0, Item const* params=0 );

   /**
    * Starts a context that is ready to run.
    *
    * The context should have been already created and configured,
    * eventually as a part of a context group, to call a function at
    * top of its code/call stack.
    */
   void addReadyContext(VMContext* ctx);

   /** Returns the result of the evaluation.
    This is actually the topmost value in the stack of the main context.
    */
   const Item& result() const;
   Item& result();

   /**
    * Wait for the process to be complete.
    *
    * This call blocks until the timeout is expired, or until the process
    * is complete. If the process completes before the call is entered,
    * the method returns immediately.
    *
    * If the process is terminated with error, the error is
    * immediately thrown right after the wait is complete.
    *
    * After a successful wait, the process result can be obtained
    * by invoking the result() method.
    */
   InterruptibleEvent::wait_result_t wait( int32 timeout=-1 );

   /**
    * Interrupts a wait for the process completion.
    *
    * If a thread is currently held in wait for this process completion,
    * the wait function returns immediately with an appropriate "wait canceled"
    * return value.
    */
   void interrupt();

   /** This is called back by the main context when done.
    This will make wait non-blocking, or wakeup waits with success.
   */
   void completed();

   /** Declares that the process was terminated by an uncaught error.
    This will make wait non-blocking, or wakeup waits with success.
   */
   void completedWithError( Error* error );

   /** Returns true if this process has been terminated.

      This is used by the virtual machine to interrupt
      any operation on any context belonging to this process.
    */

   bool terminated() const { return atomicFetch(m_terminated); }

   int32 getNextContextID();

   /**
    * Prepare the process with an entry point.
    * \return The entry point function.
    *
    * After calling this method, the process invoker can
    * put instructions to be executed in the main context,
    * and then invoke launch() as the code is complete.
    *
    * Alternatively, the caller of this method can synthesize
    * a syntactic tree in the returned syntactic function pointer,
    * but that will make this process to be usable just once.
    *
    * @note A void return PStep is automatically added at bottom of the
    * code stack to ensure a clean termination of the entry point
    * function.
    */
   SynFunc* readyEntry();

   /**
    * Returns the process termination error.
    * \return A valid error if the process was terminated with an error.
    *
    * In case the process is terminated with an error, this method will return
    * a valid error pointer.
    *
    * If this method returns 0 after the wait for process completion is complete,
    * then the process terminated correctly.
    */
   Error* error() const { return m_error; }

   void setResult( const Item& value );

protected:
   Process( VMachine* owner, bool added );
   virtual ~Process();

   void launch();
   bool checkRunning();

   friend class VMachine;
   int32 m_id;
   VMachine* m_vm;
   VMContext *m_context;
   InterruptibleEvent m_event;
   SynFunc* m_entry;
   atomic_int m_terminated;

   bool m_running;
   Mutex m_mtxRunning;

   atomic_int m_ctxId;
   Error* m_error;
   bool m_added;

   Item m_result;
   GCLock* m_resultLock;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(Process)
};

}
#endif

/* end of process.h */
