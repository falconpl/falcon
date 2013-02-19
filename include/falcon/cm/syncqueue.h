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
   SharedSyncQueue( const Class* owner );
   virtual ~SharedSyncQueue();

   void push( const Item& itm );
   bool pop( Item& target );

   void gcMark( uint32 mark );
   uint32 currentMark() const;

   void release();
   bool empty() const;

   virtual void signal( int32 count = 1);
   virtual int32 consumeSignal( int32 count = 1 );
protected:
   virtual bool lockedConsumeSignal();

private:
   class Private;
   Private* _p;
};

/*#
  @class SyncQueue
  @brief Multiple producers/consumers queue
  @ingroup parallel

   A synchronized FIFO queue.
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

     This is true if the queue is acquired.
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
     @optparam timeout Time out for the item to be popped.
     @optoaram onEmpty Returned if the queue is empty.

     @note this is a wait-point. Invoking this method forces the releasing of the queue.
     use @a SyncQueue.tryPop after a succesful wait to pop items.
    */
   FALCON_DECLARE_METHOD( pop, "timeout:[N], onEmpty:[X]" );


   /*#
     @method tryPop SyncQueue
     @brief Tries to remove an item from the queue atomically if possible.
     @optoaram onEmpty Returned if the queue is empty.

     Invoking this method doesn't relase an acquired queue.
    */
   FALCON_DECLARE_METHOD( tryPop, "onEmpty:[X]" );


   /*#
     @method wait SyncQueue
     @brief Wait the queue to be non-empty.
     @optparam timeout Milliseconds to wait for the barrier to be open.
     @return true if the barrier is open during the wait, false if the given timeout expires.

     If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
     the wait exits immediately.

     When the wait is successful, the queue is acquired exclusively, and the acquirer can pop
     as many items as desired through the @a Syncqueue.tryPop method.

     Use @a Syncqueue.release to explicitly release a queue acquired via this @b wait method.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );


   /*#
     @method release SyncQueue
     @brief Release a queue acquired through a wait operation.
     @raise AccessError if the queue is not currently acquired.
    */
   FALCON_DECLARE_METHOD( release, "" );


   FALCON_DECLARE_INTERNAL_PSTEP(AfterPop);

};

}
}

#endif	

/* end of syncqueue.h */
