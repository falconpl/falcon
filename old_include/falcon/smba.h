/*
   FALCON - The Falcon Programming Language.
   FILE: smba.h

   Small Memory Block Allocator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Dec 2008 14:42:24 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Small Memory Block Allocator.
*/

#ifndef FALCON_SMBA_H
#define FALCON_SMBA_H

#include <falcon/setup.h>

namespace Falcon
{

class Mutex;

/** Small Memory Block Allocator.
   
   This is an optimized allocator which works with fixed size small memory
   blocks: 8, 16, 32 and 64 bytes respectively, without memory overhead
   for accounting.
   
   The allocator manages directly heap pages and is able to determine the size
   of the allocated block by getting the address of the page from which it came
   from.
   
   This allocator is useful to account very small data as the trie memory blocks,
   but can be cool also to store small strings, item references and all those
   things for which you may want a lightning fast memory allocator and zero
   memory overhead.
   
   The engine provides a single SMBA for all the falcon.
*/

class SmallMemBlockAlloc
{
protected:

   typedef struct tag_Page
   {
      struct tag_Page *next;
      struct tag_Page *prev;
      void* firstFree;  // for blank allocations
      short int allocated;
      short int pageArea;
   } PAGE_HEADER;
   
   enum{
       page_list_size = 4
   };
   
   /** Pages, organized for block size */
   PAGE_HEADER* page_lists[page_list_size];
   
   /** List of free pointers, organized for block size */
   void* page_free_lists[page_list_size];
   
   // Using a pointer here because I don't want to include mt.h here.
   Mutex *m_mtx;
   
   PAGE_HEADER* newPage( int blockSize );

public:
   SmallMemBlockAlloc();
   ~SmallMemBlockAlloc();
   
   /** Allocates a small block wide at least bytes. 
      If the requested memory size is greater than the maximum size managed by this
      memory manager, function returns 0.
      
      \param bytes Quantity of memory required.
      \return The allocated block.
   */
   void* alloc( unsigned int bytes );
   
   /** Frees the given memory block.
      
      \param The memory block to be freed. 
   */
   void free( void* bytes );
};

extern SmallMemBlockAlloc SMBA;
}

#endif

/* end of smba.h */
