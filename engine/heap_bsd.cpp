/*
   FALCON - The Falcon Programming Language.
   FILE: heap_bsd.cpp

   Class for heap management - system specific for BSD compliant
   systems.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mer 14 Gen 2009 20:26:53 CET

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Class for heap management - system specific for POSIX compliant
   systems.
*/

#include <falcon/heap.h>
#include <falcon/fassert.h>

#include <unistd.h>
#include <sys/mman.h>

namespace Falcon
{

static int s_pageSize;

HeapMem Heap;

HeapMem::HeapMem()
{
   s_pageSize = getpagesize();
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
   void *ret = mmap(((void *)0), s_pageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
   fassert( ret != MAP_FAILED );
   return ret;
}

void sys_freePage( void *page )
{
   munmap( page, s_pageSize );
}

}
}

/* end of heap_bsd.cpp */
