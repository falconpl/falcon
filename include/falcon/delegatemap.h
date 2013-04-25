/*
   FALCON - The Falcon Programming Language.
   FILE: delegatemap.h

   Falcon core module -- delegate map
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Apr 2013 21:06:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DELEGATEMAP_H
#define FALCON_DELEGATEMAP_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/mt.h>
#include <falcon/string.h>

namespace Falcon {
class Item;

/** A simple opaque collection of items delegated to respond to summon messages.

  This is used in the engine and can be used by external Classes to implement
  low level instances that respond to delegation.
 */
class FALCON_DYN_CLASS DelegateMap
{
public:
   DelegateMap();
   ~DelegateMap();

   /** Sets the delegate for a given message.
    \param msg the delegate for that message.
    \param target The target for the message.

    If the message is '*', the given item is set as the general target.
    */
   void setDelegate( const String& msg, const Item& target );

   /** Sets the delegate for a given message.
    \param msg The message to be delegated.
    \param item The delegated entity for that message.
    \return The delegated item or 0 if none.

    If \b msg is '*', the target is set as delegate for any
    message.
    */
   bool getDelegate( const String& msg, Item& target ) const;

   /** Removes the delegation to the given message.
    \param msg The delegated message to be cleared.

    If the message is '*', this clears the general delegate
    and all the message delegates.
    */
   void clearDelegate( const String& msg );

   /** Remove all the delegations.

    This removes the general delegation and clears the message map
    */
   void clear();

   /** Mark all the delegated items. */
   void gcMark(uint32 mark);

   uint32 gcMark() const{ return m_mark; }
private:

   uint32 m_mark;

   mutable Mutex m_mtx;
   bool m_bHasGeneral;
   class Private;
   Private* _p;
};

}

#endif

/* delegatemap.h */
