/*
   FALCON - The Falcon Programming Language.
   FILE: vmmsg.h

   Asynchronous message for the Virtual Machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Feb 2009 16:08:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_VMMSG_H
#define FLC_VMMSG_H

/** \file
   Asynchronous message for the Virtual Machine.
*/

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/string.h>
#include <falcon/item.h>

namespace Falcon {

class GarbageLock;
class VMachine;
class Error;

/** Asynchronous message for the Virtual Machine.

   When the virtual machine receives this, it executes a broadcast loop on a coroutine
   as soon as the message reaches the main VM loop.

   Once done, the message owner is notified back via the onMessageComplete callback
   directly in the running VM thread.

   All the items used as parameters are garbage locked when given to the
   message, and garbage-unlocked on message destruction (which happens after
   the completion notify callback is called).
*/

class FALCON_DYN_CLASS VMMessage: public BaseAlloc
{
   String m_msg;
   SafeItem *m_params;
   uint32 m_allocated;
   uint32 m_pcount;
   VMMessage *m_next;
   Error* m_error;

public:
   /** Creates a VMMessage without parameters.
   \param msgName the name of the message.
   */
   VMMessage( const String &msgName );

   virtual ~VMMessage();

   /** Returns the name of the message. */
   const String& name() const { return m_msg; }

   /** Adds a paramter to the message.
   \param itm The item to be added (will be copied and garbage locked).
   */
   void addParam( const SafeItem &itm );

   /** Gets the number of parameters allocated in this message. */
   uint32 paramCount() const {return m_pcount;}

   /** Gets the nth parameter of this message. */
   SafeItem *param( uint32 p ) const;

   /** Called by the target VM when the message has been processed.
      The caller should create a subclass of VMMessage in case it
      needs to be notified about message completion and analyze
      asynchronous processing results.

      The base class implementation does nothing.

      The bProcessed parameter is set to true if at least one subscriber
      received the message, while it is set to false if the given VM hasn't the
      required slot, or if the slot is currently not subscribe by any listener.

      \note This call happens in the target VM thread.
      \param bProcessed true if called after a complete processing, false if the target VM didn't have
         active slots for this message.
   */
   virtual void onMsgComplete( bool bProcessed );

   /** Adds a message to be processed after this one.
      This method is called by the target VM to store an incoming message
      at the end of the message queue, but it may be also used by the
      message sender to send more than one message in one spot to the
      target VM.
   */
   void append( VMMessage *msg ) { m_next = msg; }

   /** Gets the next message to be processed after this one.
      Should be called only by the target VM.
   */
   VMMessage *next() const { return m_next; }
};

}

#endif

/* end of vmmsg.h */
