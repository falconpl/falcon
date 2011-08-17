/*
   FALCON - The Falcon Programming Language.
   FILE: garbagepointer.h

   Poiter that can be used to automatically dispose inner FalconData
   items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ott 15 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Poiter that can be used to automatically dispose inner FalconData
   items.
*/

#ifndef FALCON_GARBAGE_POINTER_H
#define FALCON_GARBAGE_POINTER_H

#include <falcon/garbageable.h>
#include <falcon/falcondata.h>


namespace Falcon {

class VMachine;

/** Implements a generic garbage shell for inner data.

   This pointer can be used to wrap FalconData derived classes into
   garbage sensible behavior.

   In this way, it's possible to bless simple pointers to FalconData
   managed internally by VM or inner routines, and let them to live
   in the wild. They will be marked when reachable and disposed
   cleanly when not reachable anymore.

   GarbagePointer can be set directly into items (they are the
   "user pointer" items).
*/

class FALCON_DYN_CLASS GarbagePointer: public Garbageable
{
   FalconData *m_ptr;

public:
   /** Creates the garbage pointer.
      Must be filled with the data guarded falcon data
   */
   GarbagePointer( FalconData *p ):
      Garbageable(),
      m_ptr(p)
   {
      if ( p->isSequence() )
         static_cast<Sequence*>(p)->owner( this );
   }

   /** Destructor.
      The guard will destroy its content with it.
   */
   virtual ~GarbagePointer() {}
   virtual bool finalize() { delete m_ptr; return false; }

   /** Returns the inner data stored in this garbage pointer. */
   FalconData *ptr() const { return m_ptr; }

   virtual void gcMark( uint32 gen ) {
      if( mark() != gen )
      {
         mark( gen );
         m_ptr->gcMark( gen );
      }
   }
};

}

#endif

/* end of garbagepointer.h */
