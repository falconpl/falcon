/*
   FALCON - The Falcon Programming Language.
   FILE: barrier.h

   Falcon core module -- Barrier shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_BARRIER_H
#define FALCON_CORE_BARRIER_H

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

class FALCON_DYN_CLASS SharedBarrier: public Shared
{
public:
   SharedBarrier( ContextManager* mgr, const Class* owner, bool isOpen = false );
   virtual ~SharedBarrier();

   virtual int32 consumeSignal( VMContext*, int32 count = 1 );
   void open();
   void close();

protected:
   virtual int32 lockedConsumeSignal( VMContext*, int32 count );

private:
   atomic_int m_status;
};

/*#
  @class Barrier
  @brief A barrier that can be either open or closed.
  @param status Initial status (true= open); closed by default.
  @ingroup parallel

  When a barrier is open, any wait on it will succeed; when it is closed,
  it will be blocked till opened.

  The barrier can be opened by a thread invoking the @a Barrier.open,
  and closed via @a Barrier.close.

  @note Barriers are particularly useful to signal permanent events,
  as the request for an agent to terminate.

 */
class FALCON_DYN_CLASS ClassBarrier: public ClassShared
{
public:
   ClassBarrier();
   virtual ~ClassBarrier();

   //=============================================================
   //
   virtual void* createInstance() const;

   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

private:
   /*#
     @property isOpen Barrier
     @brief Checks if the barrier is open in this moment.
    */
   FALCON_DECLARE_PROPERTY( isOpen );

   /*#
     @method open Barrier
     @brief opens the barrier.

     Opening an already open barrier has no effect; the first @a Barrier.close
     call will close the barrier, no matter how many open are issued.
    */
   FALCON_DECLARE_METHOD( open, "" );

   /*#
     @method close Barrier
     @brief Closes the barrier.

     Closing the barrier will cause any agent waiting on the barrier
     from that moment on to be blocked.
    */
   FALCON_DECLARE_METHOD( close, "" );

   /*#
     @method wait Barrier
     @brief Wait for the barrier to be open.
     @optparam timeout Milliseconds to wait for the barrier to be open.
     @return true if the barrier is open during the wait, false if the given timeout expires.

     If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
     the wait exits immediately.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );

};

}
}

#endif	

/* end of barrier.h */
