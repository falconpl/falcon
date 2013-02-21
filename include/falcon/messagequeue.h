/*
   FALCON - The Falcon Programming Language.
   FILE: messagequeue.h

   Queue of pending messages for multiple receivers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 20 Feb 2013 10:32:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MESSAGEQUEUE_H_
#define _FALCON_MESSAGEQUEUE_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/shared.h>
#include <falcon/string.h>
#include <falcon/vmcontext.h>

namespace Falcon {
class VMachine;
class Class;
class Item;

/**

 */
class FALCON_DYN_CLASS MessageQueue: public Shared
{
public:
   MessageQueue( ContextManager* mgr, const String& name="" );
   virtual MessageQueue* clone() const;

   virtual void gcMark( uint32 n );

   const String& name() const { return m_name; }

   void send( const Item& message );
   void sendEvent( const String& eventName, const Item& message );

   bool subscribe(VMContext* ctx);
   bool unsubscribe(VMContext* ctx);

   bool get( VMContext* ctx, Item& msg );
   bool peek( VMContext* ctx, Item& msg );

   bool getEvent( VMContext* ctx, String& event, Item& msg );

   virtual int32 consumeSignal( VMContext* target, int32 count = 1 );

   int32 subscribers() const;

   Shared* subscriberWaiter( int count );

   virtual void onWaiterWaiting(VMContext* ctx);

protected:
   virtual ~MessageQueue();
   virtual void onWakeupComplete();

   virtual int32 lockedConsumeSignal( VMContext* target, int32 count = 1 );

private:

   class CtxWeakRef: public VMContext::WeakRef
   {
   public:
      CtxWeakRef( MessageQueue* owner );
      virtual ~CtxWeakRef();

      virtual void onTerminate( VMContext* ctx );

   private:
      MessageQueue* m_owner;
   }
   m_ctxWeakRef;

   friend class CtxWeakRef;

   class MQFence;
   MQFence* m_firstFence;

   class Private;
   Private *_p;

   String m_name;
   atomic_int m_subscriberCount;
   uint32 m_version;

   // Messages are stored in a simple linked list as tokens.
   class Token;
   Mutex m_mtx;
   Token* m_firstToken;
   Token* m_lastToken;

   Mutex m_mtxRecycle;
   Token* m_recycleBin;
   int32 m_recycleCount;
   Token* allocToken();
   void recycleToken( Token* token );
   void recycleTokens( Token* first, Token* last );

   static const int32 MAX_TOKEN_RECYCLE_COUNT = 256;
};

}

#endif

/* end of messagequeue.h */

