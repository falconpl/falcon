/*
   FALCON - The Falcon Programming Language.
   FILE: processor.h

   Processor abstraction in the virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 05 Aug 2012 16:17:38 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PROCESSOR_H_
#define _FALCON_PROCESSOR_H_

#include <falcon/setup.h>
#include <falcon/mt.h>
#include <falcon/scheduler.h>

namespace Falcon {

class VMachine;
class VMContext;
class Error;
class Item;
class PData;

/**
 Class representing a Virtual machine processor.

 Each processor runs in its own physical machine thread.
 */
class FALCON_DYN_CLASS Processor: public Runnable
{
public:
   Processor( int32 id, VMachine* owner );
   virtual ~Processor();
   int32 id() const { return m_id; }

   virtual void* run();
   void execute(VMContext* ctx);
   void manageEvents( VMContext* ctx, int32 &events );

   void step( VMContext* ctx );

   static Processor* currentProcessor();

   void start();
   void join();


   /** Handles an error that reached the toplevel in this processor.
    */
   void onError( Error* e );

   /** Handles a raised item error that reached the toplevel in this processor.
    */
   void onRaise( const Item& item );

   /**
    Context currently being run by the processor.

    \note this method is meant to be synchronously called by the same
    thread running the ::run() method of the processor.
    */
   VMContext* currentContext() const { return m_currentContext;  }

   /** Get the Processor-wide persistent data. */
   PData* pdata() const { return m_pdata; }

private:
   class Private;
   Private* _p;

   int32 m_id;
   VMachine* m_owner;
   SysThread* m_thread;

   VMContext* m_currentContext;

   static ThreadSpecific m_me;

   Scheduler::Activity* m_activity;
   PData* m_pdata;

   static void onTimesliceExpired( void* data, Scheduler::Activity* activity );

};

}

#endif

/* end of processor.h */

