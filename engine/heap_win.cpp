/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: heap_win.cpp

   Initialization of heap_windows variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-11-01 10:20+0100UTC

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of windows specific heap.
*/

#include <falcon/heap.h>
#include <falcon/fassert.h>
#include <windows.h>

namespace Falcon
{

#define WIN_PAGE_SIZE 4096
static int s_pageSize;
static HANDLE s_heapHandle = 0;

#if 0
HeapMem Heap;

HeapMem::HeapMem()
{
   int   x = 0, y = 1, z = 2;
   s_pageSize = WIN_PAGE_SIZE;
   //s_heapHandle = GetProcessHeap();
   s_heapHandle = 0;
}

HeapMem::~HeapMem()
{
}

namespace Sys {

int sys_pageSize()
{
   return s_pageSize;
}

void *sys_allocPage()
{
   void *ret = HeapAlloc( s_heapHandle, 0, s_pageSize );
   fassert( ret != 0 );
   return ret;
}

void sys_freePage( void *page )
{
   HeapFree( s_heapHandle, 0, page );
}
}
#endif
}


/* end of heap_win.cpp */
