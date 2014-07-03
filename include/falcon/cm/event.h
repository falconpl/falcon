/*
   FALCON - The Falcon Programming Language.
   FILE: event.h

   Falcon core module -- Single signal events.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_EVENT_H
#define FALCON_CORE_EVENT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>
#include <falcon/atomic.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS SharedEvent: public Shared
{
public:
   SharedEvent( ContextManager* mgr, const Class* owner, bool isSet );
   virtual ~SharedEvent();

   virtual int32 consumeSignal( VMContext*, int32 count = 1 );

   void set();

   //Need to do something about this
   int m_status;

protected:
   virtual int32 lockedConsumeSignal(VMContext*, int32 count );
};

/*#
  @class Event
  @brief Waitable event.
  @ingroup parallel

  Events are a synchronization device that can be used by a worker
  to be notified about a status change on a shared resource. They are suitable
  to indicate changes in single worker multiple producer patterns.

  The Event doesn't support acquisition semantic, and can't be used
  to protect a resource from contemporary accesses; it is a mean to
  signal that something new has happened and requires attention.

  @note The @a Barrier synchronization primitive provides the same
  functionality as a Microsoft Windows SDK  Event with manual reset.

 */
class FALCON_DYN_CLASS ClassEvent: public ClassShared
{
public:
   ClassEvent();
   virtual ~ClassEvent();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;
};

}
}

#endif

/* end of event.h */
