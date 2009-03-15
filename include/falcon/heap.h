/*
   FALCON - The Falcon Programming Language.
   FILE: heap.h

   
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Dec 2008 13:45:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_HEAP_H
#define FALCON_HEAP_H

namespace Falcon {

namespace Sys {
   int sys_pageSize();
   void* sys_allocPage();
   void sys_freePage( void *page );
}

/**
   HeapMem class.
   
   This class allows to manage direct memory pages from the system.
   it's the base of the global SBA (Small Block Allocator).
   
   The engine creates a singleton instance of this class called Heap.
   All its functions are inlined to system specific sys_* function;
   this means that release build won't actually access the this-> pointer,
   and calling the methods of the singleton Heap will be exactly as
   calling system specific functions.
*/
class HeapMem
{
public:

   HeapMem();
   ~HeapMem();

   int pageSize() { return Sys::sys_pageSize(); }
   void *allocPage() { return Sys::sys_allocPage(); }
   void freePage( void *page ) { Sys::sys_freePage( page ); }
};

extern HeapMem Heap;

}

#endif

/* end of heap.h */
