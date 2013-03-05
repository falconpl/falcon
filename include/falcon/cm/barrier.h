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
#include <falcon/classes/classshared.h>

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
};

}
}

#endif	

/* end of barrier.h */
