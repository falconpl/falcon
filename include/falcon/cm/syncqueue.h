/*
   FALCON - The Falcon Programming Language.
   FILE: syncqueue.h

   Falcon core module -- Multiple producers/consumers queue
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_SYNCQUEUE_H
#define FALCON_CORE_SYNCQUEUE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {


class FALCON_DYN_CLASS SharedSyncQueue: public Shared
{
public:
   SharedSyncQueue( ContextManager* mgr, const Class* owner );
   virtual ~SharedSyncQueue();

   void push( const Item& itm );
   bool pop( Item& target );

   void gcMark( uint32 mark );
   uint32 currentMark() const;

   bool empty() const;

   virtual int32 consumeSignal( VMContext*, int32 count = 1 );

   bool isFair() const { return m_fair; }
protected:
   virtual int32 lockedConsumeSignal( VMContext*, int32 );
   SharedSyncQueue( ContextManager* mgr, const Class* owner, bool fair );

   class Private;
   Private* _p;

   bool m_fair;
   bool m_held;
};



class FALCON_DYN_CLASS FairSyncQueue: public SharedSyncQueue
{
public:
   FairSyncQueue( ContextManager* mgr, const Class* owner );
   virtual ~FairSyncQueue();

   virtual void signal( int32 count = 1 );
   virtual int32 consumeSignal( VMContext*, int32 count = 1 );

protected:
   virtual int32 lockedConsumeSignal( VMContext*, int32 );
};

/*#
  @class SyncQueue
  @brief Multiple producers/consumers queue
  @param fairness If true, the queue operates in fair mode.
  @ingroup parallel

   A synchronized FIFO queue, suitable for multiple producer/
   multiple consumer patterns.

   When in unfair mode, it is possible to try popping an
   item from the queue even if empty. Pushes will cause all
   the waiters to be notified, and they will have a chance to
   try to get items from the queue.

   For instance, if agent A pushes a single item on the
   queue, and agents B and C are waiting in that moment they
   get notified and might wake up at the same time. If this happens,
   only one of them will be able to get the pushed object; the other
   will get a spurious read (that can be identified by passing a
   token object to the @a SyncQueue.pop method). Also, it is possible
   that an agent D which wasn't waiting on the queue is able to
   casually pop the item posted by A, leaving both B and C without a
   valid item to be read.

   In fair mode, the queue has acquire semantic. The successful waiter
   will receive the queue in a critical section; a single pop will signal
   the queue again, and other waiting agents will be notified and given
   the queue to pop exactly one item. In fair mode, it is granted that:
   - At every wakeup it is possible to pop exactly one object; and
   - Agents that didn't successfully wait for the queue cannot pop from it; and
   - The queue is acquired in a critical section;
   - The critical section is exited explicitly and atomically by invoking the
      @a SyncQueue.pop method.

   Although this structure allows to implement any pattern of single/multiple
   consumer/producer, the unfair mode is more suitable for single consumer
   patterns; at wakeup, there will be one or more items that can be popped
   in a tight loop, while the queue is not empty; in this mode, it is not
   necessary to wait again on the queue to acquire it and then be able to
   perform a single pop.

   The fair mode is more suited for multiple consumer patterns, especially
   if more resources are polled through the same waiter.

   @note Any agent can push items in the queue, regardless of fairness or
   acquisition.
 */
class FALCON_DYN_CLASS ClassSyncQueue: public ClassShared
{
public:
   ClassSyncQueue();
   virtual ~ClassSyncQueue();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;
};

}
}

#endif	

/* end of syncqueue.h */
