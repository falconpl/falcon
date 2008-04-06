/*
   FALCON - The Falcon Programming Language.
   FILE: flc_garbageable.h

   Garbageable interface definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven dic 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Garbageable interface.

   This file contains the interface for objects and classes that can be subject
   of garbaging (i.e. because they can be inserted into items) and other utility
   definition for garbage collecting process.
*/

#ifndef flc_garbageable_H
#define flc_garbageable_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/destroyable.h>

namespace Falcon {

class VMachine;

class FALCON_DYN_CLASS Garbageable: public Destroyable
{
   VMachine *m_origin;
   uint32 m_gcSize;
   unsigned char m_gcStatus;
   Garbageable *m_garbage_next;
   Garbageable *m_garbage_prev;

   friend class MemPool;

protected:
   void updateAllocSize( uint32 nSize );

public:
   Garbageable( VMachine *vm, uint32 size=0 );

   /** Copy constructor.
      This constructor is actaully here to prevent field copy to take place:
      it just sets the m_added field to false.
   */
   Garbageable( const Garbageable &other );

   virtual ~Garbageable() {}

   void mark( byte mode ) {
      m_gcStatus = mode;
   }

   /** Return the current GC mark status. */
   unsigned char mark() {
      return m_gcStatus;
   }


   Garbageable *nextGarbage() const { return m_garbage_next; }
   Garbageable *prevGarbage() const { return m_garbage_prev; }
   void nextGarbage( Garbageable *next ) { m_garbage_next = next; }
   void prevGarbage( Garbageable *prev ) { m_garbage_prev = prev; }

   VMachine *origin() const { return m_origin; }
};

}

#endif

/* end of flc_garbageable.h */
