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
#include <falcon/classes/classshared.h>

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
};

}
}

#endif

/* end of fence.h */
