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
#include <falcon/classes/classuser.h>
#include <falcon/classes/classshared.h>

#include <falcon/method.h>
#include <falcon/atomic.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS SharedEvent: public Shared
{
public:
   SharedEvent( ContextManager* mgr, const Class* owner, bool isSet );
   virtual ~SharedEvent();

   virtual int32 consumeSignal( int32 count = 1 );

   void set();

protected:
   virtual int32 lockedConsumeSignal(int32 count );

private:
   atomic_int m_status;
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

private:

   /*#
     @method set Event
     @brief Sets the event

    */
   FALCON_DECLARE_METHOD( set, "" );


   /*#
     @method tryWait Event
     @brief Check if the event is signaled.
     @return true if the event is signaled, false otherwise.

     The check eventually resets the event if it's currently signaled.
    */
   FALCON_DECLARE_METHOD( tryWait, "" );

   /*#
     @method wait Event
     @brief Waits until the event is set.
     @optparam timeout A number of milliseconds to wait for the event to be posted.
     @return True if the wait was succesful, false if the timeout expired without
        the event being signaled.

     If @b timeout is less than zero, the wait method waits forever. If it's
     zero, it's equivalent to @a Event.tryWait.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );

};

}
}

#endif

/* end of event.h */
