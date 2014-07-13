/*
   FALCON - The Falcon Programming Language.
   FILE: eventcurrier.h


   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EVENTCOURIER_H_
#define _FALCON_EVENTCOURIER_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/gclock.h>
#include <falcon/error.h>
#include <falcon/pstep.h>

namespace Falcon {

/** Main class for embedding support.
 *
 * The event courier is a class that can be used to integrate
 * scripts and host applications, under a model that is called
 * "Event-driven embedding", or EDE.
 *
 * The EDE model consists in having the scripts to respond to events
 * generated in the host application, perform some processing, and then
 * (possibly) return some result to the host application.
 *
 * This class is also exposed to scipts, making possible to generate the same
 * messages that can be generated from an host application also from other processes
 * in the same virtual machine, or even from other contexts in the same process.
 *
 * The class has support to specially configure the contexts so that the scripts
 * require no particular level of integration; however, the class also provides
 * support for the script to interact with the integration, so that the callbacks
 * for the events can be determined by the script themselves.
 *
 */
class FALCON_DYN_CLASS EventCourier
{
public:
   EventCourier();
   virtual ~EventCourier();

   /** Create an event group.
    *
    * Event groups can be used to deliver multiple events to the same callback.
    */
   void addToGroup( int64 groupID, int64 eventID);
   void removeFromGroup( int64 groupID, int64 eventID);

   void setCallback( int64 id, const Item& callback );
   void clearCallback( int64 id );

   void prepareContext( VMContext* ctx );
   void terminate();

   /** Token returned by the sendEvent request.
    *
    * Notice: tokens are allocated in a pool in the event courier class.
    */
   class FALCON_DYN_CLASS Token {
   public:
      Token(EventCourier* owner);
      bool wait(int64 to = -1);

      Item& result() const;
      Error* error() const;

      void incref();
      void decref();

      int64 eventID() const { return m_evtID; }
      ItemArray& params() { return m_params; }
      const ItemArray& params() const { return m_params; }

      const Item& result() const;

      void prepare( int64 id, Item* params, int32 pcount, Event* evt);
      void prepare( int64 id, Item* params, int32 pcount, Shared* sh);
      void prepare( int64 id, Item* params, int32 pcount );

      void complete( const Item& result );
      void abort( Error* e );

   private:
      ~Token();
      void dispose();

      ItemArray m_params;

      atomic_int m_refcount;
      GCLock* m_gcLockResult;
      int64 m_evtID;
      EventCourier* m_owner;

      Event* m_event;
      Shared* m_shared;

      friend class EventCourier;
   };

   Token* sendEvent( int64 id, Item* params, int pcount, bool useShared = false );

   void release( Token* token );

private:
   int64 m_lastGroupID;
   PStep* m_pstepHandleEvents;

   class Private;
   Private* _p;
};

}

#endif /* EVENTCOURIER_H_ */

/* eventcourier.h */
