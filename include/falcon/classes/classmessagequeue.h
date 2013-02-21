/*
   FALCON - The Falcon Programming Language.
   FILE: classmessagequeue.h

   Message queue reflection in scripts
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 20 Feb 2013 14:15:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSMESSAGEQUEUE_H
#define FALCON_CLASSMESSAGEQUEUE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

#include <falcon/method.h>
#include <falcon/property.h>

namespace Falcon {

/*#
 @class MessageQueue
 @brief Multiple receivers message queue.

*/
class FALCON_DYN_CLASS ClassMessageQueue: public ClassShared
{
public:
   ClassMessageQueue();
   virtual ~ClassMessageQueue();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

private:

   /*#
     @property subscribers MessageQueue
     @brief Number of subscribers currently listening on this queue.

    As the message queue doesn't support late subscribers, a sender
    can safely check for subscribers count to be > 0 before preparing
    a complex and time-consuming message that would be immediately
    discarded by the queue.

    */
   FALCON_DECLARE_PROPERTY( subscribers );

   /*#
     @property empty MessageQueue
     @brief True if the queue has not any item that can be read by the calling context.

    */
   FALCON_DECLARE_PROPERTY( empty );

   /*#
     @method send MessageQueue
     @brief Sends a new message to all the subscribers
     @optparam message An arbitrary item that will be received by all the subscribers.

      The parameter @b count must be greater or equal to 1.
    */
   FALCON_DECLARE_METHOD( send, "message:X,..." );

   /*#
     @method get MessageQueue
     @brief Gets a message waiting on the queue.
     @return the item posted by send()
     @raise AccessError if the queue has not any new message for us.

     Removes the message from the front of the queue.
    */
   FALCON_DECLARE_METHOD( get, "" );

   /*#
     @method peek MessageQueue
     @brief Peek a message waiting on the queue.ClassMessageQueue
     @return the item posted by send()
     @raise AccessError if the queue has not any new message for us.

     Reads the message in front of the queue, without removing it.
    */
   FALCON_DECLARE_METHOD( peek, "" );

   /*#
     @method subscribersFence MessageQueue
     @brief Return a fence that gets notified when the required amount of subscribers have joined the queue.
     @return A waitable fence.

     This facility obviates the need to use other synchronization devices to
     get to know when the subscribers that are expected to join have really
     joined.

     The message queue doesn't implement late-arrival messaging semantic. Messages
     are delivered exclusively to the subscribers that have subscribed the list
     in the moment the message is sent.

     For this reason, before sending critical messages to listeners that might
     be not ready to receive them, it is necessary to ascertain that they have
     actually subscribed. This could be done using other synchronization devices,
     but the subscribersFence() method returns a fence that is automatically
     updated as the receivers subscribe the queue, so no other action is required
     on the subscribers side.
    */
   FALCON_DECLARE_METHOD( subscribersFence, "count:N" );


   /*#
     @method tryWait MessageQueue
     @brief Check if there are new messages for us.
     @return true if there are new messages for us.
    */
   FALCON_DECLARE_METHOD( tryWait, "" );

   /*#
     @method wait MessageQueue
     @brief Wait for new messages to be available.
     @optparam timeout Milliseconds to wait for the barrier to be open.
     @return true if the barrier is open during the wait, false if the given timeout expires.

     If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
     the wait exits immediately.

     @note If not previously subscribed, the receiver gets subscribed
     at the moment the wait is entered.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );
};

}

#endif	

/* end of classmessagequeue.h */
