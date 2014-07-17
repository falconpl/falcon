/*
   FALCON - The Falcon Programming Language.
   FILE: eventcurrier.h


   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EVENTCOURIER_H_
#define _FALCON_EVENTCOURIER_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/gclock.h>
#include <falcon/error.h>
#include <falcon/pstep.h>
#include <falcon/mt.h>
#include <falcon/itemarray.h>

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

   const static length_t DEFAULT_POOL_SIZE = 32;

   /** Create an event group.
    *
    * Event groups can be used to deliver multiple events to the same callback.
    */
   /*
   void addToGroup( int64 groupID, int64 eventID);
   void removeFromGroup( int64 groupID, int64 eventID);
   */

   void setCallback( int64 id, const Item& callback, const String& msg = "" );
   void clearCallback( int64 id );

   void prepareContext( VMContext* ctx );
   void terminate();

   /** Callback functor used to notify completion of event management.
    *
    */
   class Callback
   {
   public:
      virtual ~Callback(){}
      virtual void operator()(int32 eventID, const Item& result, Error* error = 0 ) = 0;
   };

   /** Token returned by the sendEvent request.
    *
    * Notice: tokens are allocated in a pool in the event courier class.
    */
   class FALCON_DYN_CLASS Token {
   public:
      Token(EventCourier* owner);
      /** Wait for an event-based token to be completed, and eventually returns a result.
       * \param result An item where to place the operation result.
       * \param to A time out in milliseconds to wait for a result.
       * \return true if the operation is completed, false if it's still pending.
       *
       * If the operation terminated with an error, the function will rethrow the error
       * in the waiting context.
       */
      bool wait(Item& result, int64 to = -1);

      const Item& result() const;
      Error* error() const { return m_error; }

      void incref();
      void decref();

      int64 eventID() const { return m_evtID; }
      ItemArray& params() { return m_params; }
      const ItemArray& params() const { return m_params; }


      void prepare( int64 id, Item* params, int32 pcount, Event* evt);
      void prepare( int64 id, Item* params, int32 pcount, Shared* sh);
      void prepare( int64 id, Item* params, int32 pcount, Callback* cb);
      void prepare( int64 id, Item* params, int32 pcount );

      void completed( const Item& result );
      void aborted( Error* e );
      void unhandled();

      void gcMark( uint32 md );
      uint32 currentMark() const { return  m_mark; }

   private:

      EventCourier* m_owner;

      ItemArray m_params;

      atomic_int m_refcount;
      GCLock* m_gcLockResult;
      Error* m_error;
      int64 m_evtID;

      uint32 m_mark;

      typedef enum {
         mode_none,
         mode_event,
         mode_shared,
         mode_cb
      }
      t_mode;
      t_mode m_mode;

      union {
         Event* event;
         Shared* shared;
         Callback* cb;
      }
      m_completion;

      Item m_NIL;

      ~Token();
      void dispose();
      void clear();

      void fillParams( Item* items, int pcount );

      friend class EventCourier;
   };

   /** Send an event without waiting for completion. */
   void sendEvent( int64 id, Item* params, int pcount );

   /** Send an event and wait for completion on the given Falcon::Event.
    * \param id The event ID
    * \param params The parameters to be sent together with the event.
    * \param pcount the number of parameters in the params vector.
    * \param evt The event to be signaled at completion.
    * \return A token that can be used to receive the operation result.
    *
    * The parameter vector is copied, so it doesn't need to stay valid for
    * the duration of the duration of the event processing.
    * \note evt Ownership stays in charge of the sender.
    *  */
   Token* sendEvent( int64 id, Item* params, int pcount, Event* evt );

   /** Send an event and wait for completion on the given Falcon::Shared.
    * \param id The event ID
    * \param params The parameters to be sent together with the event.
    * \param pcount the number of parameters in the params vector.
    * \param evt The shared variable to be signaled at completion.
    * \return A token that can be used to receive the operation result.
    *
    * The parameter vector is copied, so it doesn't need to stay valid for
    * the duration of the duration of the event processing.
    *
    * \note evt Ownership stays in charge of the sender.
    *  */
   Token* sendEvent( int64 id, Item* params, int pcount, Shared* evt );

   /** Send an event and invoke the given callback at completion.
    * \param id The event ID
    * \param params The parameters to be sent together with the event.
    * \param pcount the number of parameters in the params vector.
    * \param cbt The Callback functor to be invoked at completion.
    * \return A token that can be used to receive the operation result.
    *
    * \note cbt Ownership stays in charge of the sender.
    *  */
   void sendEvent( int64 id, Item* params, int pcount, Callback* cbt );

   /** Gets the next event queued on this courier. */
   Token* popEvent() throw();

   Shared* eventPosted() const;

   /** Gets the item handler for the given event, if any.
    * \param eventID The event that should be listening to this message.
    * \param tgtMessage if not null on output, the handler is to be invoked via summon.
    * \return 0 if there isn't any registered handler for that event, a valid Item* if there is a handler.
    *
    */
   const Item* getHandler( int64 eventID, String& tgtMessage ) const;
   const Item* getDefaultHandler() const;
   void setDefaultHandler( const Item& df );

   void gcMark(uint32 mark);
   uint32 currentMark() const { return m_mark; }

   /** Subscribe a direct callback.
    *
    */
   void subscribe( int64 eventID, const Item& handler );

   /** Subscribe via summoning messages.
    *
    */
   void subscribe( int64 eventID, const Item& handler, const String& message );

   /** Unsubcribe from an event.
    *
    */
   void unsubscribe( int64 eventID );

   /** This is actually called by the tokens when necessary.
    *
    */
   void release( Token* token );

   /** Get the current MAXIMUM size of the message pool.
    *
    */
   length_t poolSize() const;

   /** Set the maximum size of the message pool.
    *
    */
   void poolSize( length_t pz );

   virtual void onUnhandled( Token* tk, VMContext* ctx );

   bool throwOnUnhandled() const { return m_throwOnUnhandled;  }
   void throwOnUnhandled( bool b ) { m_throwOnUnhandled = b;  }

protected:
   Token* allocToken();

private:
   int64 m_lastGroupID;
   uint32 m_mark;

   GCLock* m_defaultHanlderLock;
   mutable Mutex m_mtxDflt;

   Shared* m_sharedPosted;
   class Private;
   Private* _p;

   bool m_throwOnUnhandled;
};

}

#endif /* EVENTCOURIER_H_ */

/* eventcourier.h */
