/*
   FALCON - The Falcon Programming Language.
   FILE: garbageable.h

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
#include <falcon/gcalloc.h>

namespace Falcon {

class FALCON_DYN_CLASS GarbageableBase: public GCAlloc
{
protected:
   GarbageableBase *m_garbage_next;
   GarbageableBase *m_garbage_prev;
   mutable uint32 m_gcStatus;

public:
   GarbageableBase() {}

   /** Copy constructor.
   */
   GarbageableBase( const GarbageableBase &other );

   virtual ~GarbageableBase();

   /** Performs pre-delete finalization of the object.
      If this function returns false, then the destructor is called.
      If it returns true, it means that the finalizer has somewhat reclaimed
      the memory in a clean way (i.e. deleting itself), so the delete on this
      garbageable won't be called.

      \return true to prevent destructor to be applied on this garbageable.
   */
   virtual bool finalize();

   /** Returns an estimation of the size occupied by this object in memory.
      The final GC size is determined by an heuristic algorithm allocating
      part of the allocated space to the items returning 0 from this call
      (the default), taking away all the memory declared by items not
      returning 0.
   */
   virtual uint32 occupation();

   void mark( uint32 gen ) const {
      m_gcStatus = gen;
   }

   /** Return the current GC mark status. */
   uint32 mark() const {
      return m_gcStatus;
   }

   GarbageableBase *nextGarbage() const { return m_garbage_next; }
   GarbageableBase *prevGarbage() const { return m_garbage_prev; }
   void nextGarbage( GarbageableBase *next ) { m_garbage_next = next; }
   void prevGarbage( GarbageableBase *prev ) { m_garbage_prev = prev; }
};


class FALCON_DYN_CLASS Garbageable: public GarbageableBase
{
public:
   Garbageable();

   /** Copy constructor.
   */
   Garbageable( const Garbageable &other );

   virtual ~Garbageable();

   /** Applies mark to subclasses.
    * By default, this method just changes the mark() value.
    *
    * Subclasses having deep data may overload this to take care
    * of marking it.
    */
   virtual void gcMark( uint32 mk );
};

}

#endif

/* end of garbageable.h */
