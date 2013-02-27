/*
   FALCON - The Falcon Programming Language.
   FILE: fence.h

   Falcon core module -- Waiter for multiple conditions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_FENCE_H
#define FALCON_CORE_FENCE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classuser.h>
#include <falcon/classes/classshared.h>

#include <falcon/method.h>
#include <falcon/atomic.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS SharedFence: public Shared
{
public:
   SharedFence( ContextManager* mgr, const Class* owner, int32 fenceCount, bool eventSemantic );
   virtual ~SharedFence();

   virtual void signal( int32 count = 1 );
   virtual int32 consumeSignal( VMContext*, int32 count = 1 );

   int32 level() const;
   int32 count() const;
   bool isEvent() const { return m_bEventSemantic; }

   void count( int32 count );

protected:
   virtual int32 lockedConsumeSignal( VMContext*, int32 count );

private:
   int32 m_level;
   int32 m_fenceCount;
   bool m_bEventSemantic;
};

/*#
  @class Fence
  @brief Synchronization device waiting for multiple events.
  @ingroup parallel
  @param count Count of signal for the fence to be waitable.
  @optparam asEvent If true, this fence has event semantic.

   Fences are synchronization devices that wait for a given count
   of signals to be sent before the receiver can proceed. For instance,
   if the fence count is set to 5, a waiter must receive 5 signals
   before it can proceed.

   They come in two flavors: with or without event semantic.

   With event semantic, signaling a fence that has already received
   the number of signals required to proceed, has no effect. A fence
   with event semantic and count 1 is equivalent to an event.

   Without event semantic, the signal count is allowed to fall below
   zero. As a signal is received, the the fence level is decremented;
   a waiter can succesfully wait on a fence whose level is zero or less,
   and when it succesfully waits, the fence count is atomically added to
   the current level.

   For instance, if a fence without event semantic, and with a count of 5,
   receives 7 signals, and then the waiter is awaken, the level is reset to 3:
   a new wait might succeed after just 3 new signals are received. If the
   same fence receives 11 signals, three waiters are allowed to proceed, and
   the wait count for the fourth waiter is set to 1.

   @note Fence has @b not the acquire semantic.
 */
class FALCON_DYN_CLASS ClassFence: public ClassShared
{
public:
   ClassFence();
   virtual ~ClassFence();

   //=============================================================
   //
   virtual void* createInstance() const;

   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

private:
   /*#
     @property level Fence
     @brief Current signal level of the fence.

     This is a read/only property.
    */
   FALCON_DECLARE_PROPERTY( level );

   /*#
     @property isEvent Fence
     @brief True if the current fence has event semantic

     This is a read/only property.
    */
   FALCON_DECLARE_PROPERTY( isEvent );

   /*#
     @property count Fence
     @brief Count of signals that this fence waits on.

     This is property can be changed at runtime, but the
     count level doesn't affect the level. In other words,
     the count change becomes effective after
     the first waiter has waken up.

     For example,
     if a fence with count 5 received 2 signals, and it is
     now at level 3, setting count to 2 won't cause the
     waiters to wake up; the fence still need to receive
     3 signals before waiters are waken up.
    */
   FALCON_DECLARE_PROPERTY( count );

   /*#
     @method signal Fence
     @brief Signals the fence
     @optparam count Count of signals to be sent to the fence.

      The parameter @b count must be greater or equal to 1.
    */
   FALCON_DECLARE_METHOD( signal, "count:[N]" );

   /*#
     @method tryWait Semaphore
     @brief Check if the semaphore is signaled.
     @return true if the semaphore is signaled, false otherwise.

     The check eventually resets the semaphore if it's currently signaled.
    */
   FALCON_DECLARE_METHOD( tryWait, "" );

   /*#
     @method wait Event
     @brief Waits until the event is set.
     @optparam timeout A timeout in milliseconds to wait for.

    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );

};

}
}

#endif

/* end of fence.h */
