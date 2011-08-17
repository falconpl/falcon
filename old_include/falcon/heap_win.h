/*
   FALCON - The Falcon Programming Language.
   FILE: flc_heap_win.h

   Windows specific class for Dynamic load system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-11-1 02:34+0200UTC

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_HEAP_WIN_H
#define flc_HEAP_WIN_H

#include <falcon/fassert.h>

/** Page size.
\todo add a configure system to put the page size in the config.h
*/
#define PAGE_SIZE    4096

#include <windows.h>

namespace Falcon
{

class HeapMem_Win32
{
   static long m_pageSize;
   static HANDLE m_heapHandle;

public:
/*
   static void init() {
   }

   static void uninit() {}
*/
   static void *getPage() { return getPages(1); }

   static void *getPages( int pages )
   {
      if ( m_heapHandle == 0 )
         m_heapHandle = GetProcessHeap();

      void *ret = HeapAlloc( m_heapHandle, HEAP_NO_SERIALIZE, pages * PAGE_SIZE );
      fassert( ret != 0 );
      return ret;
   }

   static void freePage( void *memory ) { free( memory, 1 ); } // free one page
   static void free( void *memory, int pages )
   {
      fassert( m_heapHandle != 0 );
      HeapFree( m_heapHandle, HEAP_NO_SERIALIZE, memory );
   }

   //static long pageSize() { return m_pageSize; }

   static long pageSize() { return PAGE_SIZE; }
};

typedef HeapMem_Win32 HeapMem;

}

#endif

/* end of flc_heap_win.h */
