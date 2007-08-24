/*
   FALCON - The Falcon Programming Language.
   FILE: flc_memblock.h
   $Id: memblock.h,v 1.1.1.1 2006/10/08 15:05:28 gian Exp $

   Managed block abstraction.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio ott 7 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
