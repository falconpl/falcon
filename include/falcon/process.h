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

namespace Falcon {

class VMachine;
class VMContext;

/** Process Entity.

 This entity represents an execution process in a virtual machine.

 The process has a VMContext which is the main context, it can be
 waited upon, interrupted asynchronously and checked for a result.

 The constructor is typically created by the virtual machine, which
 sets up a main context for the process and prepares it to run the
 desired code and/or symbol.
 */
class FALCON_DYN_CLASS Process
{
public:
   Process( VMachine* owner );
   Process( VMachine* owner, VMContext* mainContext );
   virtual ~Process();

   int32 id() const { return m_id; }
   void id( int32 i ) { m_id = i; }

   VMachine* vm() const { return m_vm; }
   VMContext* mainContext() const { return m_context; }

   void start( Function* main, int pcount = 0 );
   void start( Closure* main, int pcount = 0);
   void startItem( Item& main, int pcount, Item* params );

   /** Returns the result of the evaluation.
    This is actually the topmost value in the stack of the main context.
    */
   const Item& result() const;
   Item& result();

   bool wait( int64 timeout=-1 );
   void interrupt();

   /** This is called back by the main context when done.
    This will make wait non-blocking, or wakeup waits with success.
   */
   void completed();

   /** Returns true if this process has been terminated.

      This is used by the virtual machine to interrupt
      any operation on any context belonging to this process.
    */

   bool terminated() const { return atomicFetch(m_terminated); }

private:
   int32 m_id;
   VMachine* m_vm;
   VMContext *m_context;
   InterruptibleEvent m_event;
   atomic_int m_terminated;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(Process)
};

}
#endif

/* end of process.h */
