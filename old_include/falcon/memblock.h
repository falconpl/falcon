/*
   FALCON - The Falcon Programming Language.
   FILE: flc_memblock.h

   Managed block abstraction.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio ott 7 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#ifndef flc_flc_memblock_H
#define flc_flc_memblock_H

namespace Falcon {

class MemBlock
{
   MemBlock *m_prev;
   MemBlock *m_next;
public:
   MemBlock( MemBlock *prev = 0, MemBlock *next=0 ):
      m_prev(prev),
      m_next(next)
   {}
   
   void *memory() { return ( void *) ((char *)this)+sizeof(this); }
   
};

}

#endif

/* end of flc_memblock.h */
