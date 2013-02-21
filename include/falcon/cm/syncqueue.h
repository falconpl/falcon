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
#include <falcon/classes/classuser.h>
#include <falcon/classes/classshared.h>

#include <falcon/pstep.h>

#include <falcon/method.h>

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
   casually pop the item posted by A, leaving both B and C wihtout a
   valid item to be read.

   In fair mode, the queue has acquire semantic. The succesful waiter
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

private:
   /*#
     @property empty Barrier
     @brief Checks if queue is empty at the moment.

     This information has a meaning only if it can be demonstrated
     that there aren't other producers able to push data in the queue
     in this moment.

    */
   FALCON_DECLARE_PROPERTY( empty );

   /*#
     @method push SyncQueue
     @brief Pushes one or more items in the queue.
     @param item The item pushed in the queue.
     @optparam ... More items to be pushed atomically.

     It is not necessary to acquire the queue to push an item.
     Also, pushing an item does not automatically release the queue.
    */
   FALCON_DECLARE_METHOD( push, "item:X,..." );

   /*#
     @method pop SyncQueue
     @brief Removes an item from the queue atomically, or waits for an item to be available.
     @optparam onEmpty Returned if the queue is empty.
     @raise AccessError if the queue is in fair mode and the pop method is invoked without
           having acquired the resource with a successfull wait.

      In non fair mode, even if the queue is signaled and the wait operation is successful,
      there is no guarantee that the queue is still non-empty when this agent
      tires to pop the queue. The pop method is granted to return an item from
      the queue if and only if a wait operation was successful and there aren't other
      agents trying to pop from this resource.

      In fair mode, this method can be invoked only after having acquired the queue
      through a successful wait operation. It is then granted that the method will return
      an item, and the @b onEmpty parameter, if given, will be ignored.
    */
   FALCON_DECLARE_METHOD( pop, "onEmpty:[X]" );

   /*#
     @method wait SyncQueue
     @brief Wait the queue to be non-empty.
     @optparam timeout Milliseconds to wait for the barrier to be open.
     @return true if the barrier is open during the wait, false if the given timeout expires.

     If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
     the wait exits immediately.

     If the queue is in fair mode, a successful wait makes the invoker to
     enter a critical section; the pop method will then release the queue.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );
};

}
}

#endif	

/* end of syncqueue.h */
