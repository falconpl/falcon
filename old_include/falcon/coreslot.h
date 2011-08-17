/*
   FALCON - The Falcon Programming Language.
   FILE: coreslot.h

   Core Slot - Messaging slot system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Jan 2009 18:28:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORESLOT_H
#define FALCON_CORESLOT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/itemlist.h>
#include <falcon/string.h>
#include <falcon/traits.h>
#include <falcon/mt.h>
#include <falcon/falconobject.h>

namespace Falcon {

class VMachine;
class VMContext;
class VMMessage;

/** Slot for messages sent around by the VM.
   This class provide abstract support for low level messaging system.

   The slot represents an end of the communication process where the incoming
   message is definitely.
*/
class FALCON_DYN_CLASS CoreSlot: public ItemList
{
   String m_name;
   mutable Mutex m_mtx;
   mutable volatile int32 m_refcount;

   Item m_assertion;
   bool m_bHasAssert;

   Map* m_children;

public:
   CoreSlot( const String &name ):
      m_name( name ),
      m_refcount(1),
      m_bHasAssert( false ),
      m_children( 0 )
   {}

   virtual ~CoreSlot();

   const String& name() const { return m_name; }

   /** Prepares a broadcast from the current frame.

      Meant to be called from inside extension functions going to
      perform broadcasts, this function prepares the multi-call frame
      used for repeated broadcast-based calls.

      @param vmc The VM context on which the broadcast is going to be performed.
      @param pfirst The first parameter in the current call frame that must be repeated.
      @param pcount Parameter count to be passed in the broadcast.
      \param msg The message that caused the slot to be broadcast (can be none if internally broadcast).
      \param msgName Name of the message that must be broadcast if different from the name of this slot -- used by send.
   */
   void prepareBroadcast( VMContext *vmc, uint32 pfirst, uint32 pcount, VMMessage* msg = 0, String* nsgName = 0 );

   /** Remove a ceratin item from this slot.
      This will remove an item considered equal to the subscriber from this list.
   */
   bool remove( const Item &subsriber );


   void incref() const;
   void decref();

   /** Returns true if this slot is associated with an assertion. */
   bool hasAssert() const { return m_bHasAssert; }
   
   /** Return the asserted data. 
      This data is meaningless if hasAssert() isn't true.
   */
   const Item &assertion() const { return m_assertion; }
   
   /** Sets an assertion for this slot.
      No action is taken.
   */
   void setAssertion( const Item &a ) { m_assertion = a; m_bHasAssert = true; }

   /** Performs an assertion for this slot.
      Also, prepares the VM to run a broadcast loop with the asserted item.
   */
   void setAssertion( VMachine* vm, const Item &a );

   /** Retracts the assert data. 
      This function does nothing if the slot didn't have an assertion.
   */
   void retract() { 
      m_bHasAssert = false; 
      m_assertion.setNil(); // so the GC can rip it.
   }

   virtual CoreSlot *clone() const;
   virtual void gcMark( uint32 mark );

   virtual void getIterator( Iterator& tgt, bool tail = false ) const;
   virtual void copyIterator( Iterator& tgt, const Iterator& source ) const;

   virtual void disposeIterator( Iterator& tgt ) const;

   /** Gets or eventually creates a child slot.

    */
   CoreSlot* getChild( const String& name, bool create = false );
};


/** Traits for the core slots. */
class CoreSlotPtrTraits: public ElementTraits
{
public:
   virtual ~CoreSlotPtrTraits() {}
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

namespace traits
{
   extern CoreSlotPtrTraits &t_coreslotptr();
}

bool coreslot_broadcast_internal( VMachine *vm );

/** Class taking care of finalizing the core slots when they are published to scripts. */
class FALCON_DYN_CLASS CoreSlotCarrier: public FalconObject
{
public:
   CoreSlotCarrier( const CoreClass* generator, CoreSlot* cs, bool bSeralizing = false );
   CoreSlotCarrier( const CoreSlotCarrier &other );
   virtual ~CoreSlotCarrier();
   virtual CoreSlotCarrier *clone() const;

   /** Change slot after the creation of this carrier (for VMSlot_init) */
   void setSlot( CoreSlot* cs );
};

CoreObject* CoreSlotFactory( const CoreClass *cls, void *user_data, bool bDeserial );

}

#endif

/* end of coreslot.h */

