/*
   FALCON - The Falcon Programming Language.
   FILE: flc_membuf.h
   $Id: membuf.h,v 1.1.1.1 2006/10/08 15:05:40 gian Exp $

   Temporary memory buffer class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ott 9 2004
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
   This file is temporary; it will be deleted as soon as memory manager for VM
   is fully functional.
*/

#ifndef flc_flc_membuf_H
#define flc_flc_membuf_H

#include <falcon/types.h>

namespace Falcon {

typedef void (*cleanup_func)(void *data);

/** Temporary class used to hold memory buffer while waiting for the fully
    page managed memory buffers.
    This memory buffers are commonly allocated via Falcon::memAlloc(). In the final version
    the memory buffers used by items will be directly mapped into memory pages and managed
    by the memory pool manager.
*/

class MemBuf
{
   MemBuf *m_prev;
   MemBuf *m_next;
   cleanup_func m_destroyer;
   uint32 m_size;
   byte m_gcData;
   /** General purpose byte.
      This byte is left for the item contents to be used for various reasons.
      As the GC data is a byte in this area, adding a flag byte to this area
      causes no waste (in 16 or 32 bit allignment), while adding it to the
      item content would cause a waste.
   */
   byte m_general;
public:

   MemBuf( uint32 size, MemBuf *prev=0, MemBuf *next=0, cleanup_func func=0 ):
      m_size( size ),
      m_prev( prev ),
      m_next( next ),
      m_destroyer( func ),
      m_gcData(0)
   {}

   ~MemBuf() {
      if ( m_destroyer ) {
         m_destroyer( dataSpace() );
      }
      if (m_prev != 0 )
         m_prev->m_next = m_next;
      if ( m_next != 0 )
         m_next->m_prev = m_prev;
   }

   MemBuf *next() const  { return m_next; }
   void next( MemBuf *n ) { m_next = n; }
   MemBuf *prev() const  { return m_prev; }
   void prev( MemBuf *p ) { m_prev = p; }

   byte *dataSpace() { return ((byte*)this) + sizeof(MemBuf); }
   static MemBuf *defSpace( void *mem ) { return ( ((MemBuf*)mem) - 1 ); }

   void setCleanupFunction( cleanup_func func ) {
      m_destroyer = func ;
   }

   cleanup_func getCleanupFunction() {
      return m_destroyer;
   }

   void cleanup() {
      if ( m_destroyer != 0 )
         m_destroyer( this );
   }

   /** Set the current mark status. */
   void gcMark( byte mode ) {
      m_gcData = (m_gcData & 0xfe) | mode;
   }

   /** Return the current GC mark status. */
   byte gcMark() const {
      return ( m_gcData & 0x1);
   }

   uint32 size() const { return  m_size; }

   /** General purpose byte.
      This byte is left for the item contents to be used for various reasons.
      As the GC data is a byte in this area, adding a flag byte to this area
      causes no waste (in 16 or 32 bit allignment), while adding it to the
      item content would cause a waste.
   */
   byte &general() { return m_general; }
};

}

#endif

/* end of flc_membuf.h */
