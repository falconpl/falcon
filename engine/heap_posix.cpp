/*
   FALCON - The Falcon Programming Language.
   FILE: heap_posix.cpp

   Class for heap management - system specific for POSIX compliant
   systems.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Dec 2008 13:45:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

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

// For BSD
#ifndef MAP_ANONYMOUS
# define MAP_ANONYMOUS MAP_ANON
#endif

namespace Falcon
{

static int s_pageSize;

HeapMem Heap;

HeapMem::HeapMem()
{
   s_pageSize = sysconf( _SC_PAGESIZE );
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
   void *ret = mmap(((char *)0), s_pageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   fassert( ret != MAP_FAILED );
   return ret;
}

void sys_freePage( void *page )
{
   munmap( (char*) page, s_pageSize );
}

}
}

/* end of heap_posix.cpp */
