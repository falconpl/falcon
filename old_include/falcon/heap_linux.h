/*
   FALCON - The Falcon Programming Language.
   FILE: flc_heap_linux.h

   Base class for heap management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer set 29 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Heap memory for linux.
*/

#ifndef flc_HEAP_LINUX_H
#define flc_HEAP_LINUX_H

#include <unistd.h>
#include <falcon/fassert.h>

/** Page size.
\todo add a configure system to put the page size in the config.h
*/
#define PAGE_SIZE    4096


namespace Falcon {

class HeapMem_Linux
{
   static long m_pageSize;

public:
/*
   static void init() {
      if ( m_pageSize == 0 )
      {
         m_pageSize = sysconf( _SC_PAGESIZE );
      }
   }

   static void uninit() {}
*/
   static void *getPage() { return getPages(1); }

   static void *getPages( int pages )
   {
      //fassert( m_pageSize != 0 );
      void *ret = mmap(((void *)0), pages * PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      fassert( ret != MAP_FAILED );
      return ret;
   }

   static void freePage( void *memory ) { free( memory, 1 ); } // free one page
   static void free( void *memory, int pages )
   {
      //fassert( m_pageSize != 0 );
      munmap( memory, pages * PAGE_SIZE );
   }

   //static long pageSize() { return m_pageSize; }

   static long pageSize() { return PAGE_SIZE; }
};

typedef HeapMem_Linux HeapMem;

}

#endif

/* end of flc_heap_linux.h */
