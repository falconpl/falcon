/*
   FALCON - The Falcon Programming Language.
   FILE: wvmcontext.h

   Falcon virtual machine -- waitable VM context.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 11:36:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_WVMCONTEXT_H_
#define FALCON_WVMCONTEXT_H_

#include <falcon/setup.h>
#include <falcon/vmcontext.h>
#include <falcon/mt.h>

namespace Falcon {

/**
 *
 */
class FALCON_DYN_CLASS WVMContext: public VMContext
{
public:
   WVMContext( Process* prc, ContextGroup* grp=0 );
   virtual ~WVMContext();

   /** Calls and immediately send to process execution this function. */
   void start( Function* f, int32 np = 0, Item const* params=0 );

   /** Calls and immediately send to process execution this function. */
   void start( Closure* closure, int32 np=0, Item const* params=0 );

   /** Calls and immediately send to process execution this item. */
   void startItem( const Item& item, int32 np=0, Item const* params=0 );

   /** Access the event that will be signaled on operation completion. */
   Event* completeEvent() const { return m_evtComplete; }

   /** Wait for the completion event to be signaled */
   bool wait(int32 to=-1) const { return m_evtComplete->wait(to); }
   /** Wait for the completion event to be signaled */
   bool tryWait() const { return m_evtComplete->wait(0); }

   /** Callback function for context completion. */
   typedef void (*complete_cbfunc)(WVMContext* ctx, void* data);

   /** Declares a function that will be invoked when the processing is complete. */
   void setOnComplete( complete_cbfunc func, void* data=0 );

   virtual void gcPerformMark();

   /** Access the return value of the completed processing. */
   const Item& result() const { return m_result; }

   virtual void onComplete();
private:
   Event* m_evtComplete;
   complete_cbfunc m_completeCbFunc;
   void* m_completeData;

   class PStepComplete;
   friend class PStepComplete;
   PStep* m_stepComplete;

   Item m_result;
};

}

#endif /* FALCON_VMCONTEXT_H_ */

/* end of wvmcontext.h */
