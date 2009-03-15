/*
   FALCON - The Falcon Programming Language.
   FILE: smba.cpp

   Small Memory Block Allocator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Dec 2008 14:42:24 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/types.h>
#include <falcon/smba.h>
#include <falcon/heap.h>
#include <falcon/mt.h>

namespace Falcon
{

int32 s_pageSize;
int32 s_pageMask;

#if 0
SmallMemBlockAlloc SMBA;

SmallMemBlockAlloc::SmallMemBlockAlloc()
{
   for( int i = 0; i < page_list_size; i ++ )
   {
      PAGE_HEADER* p = newPage( 8 << i );
      page_lists[i] = p;
      page_free_lists[i] = 0;
   }
   
   m_mtx = new Mutex;

   // we don't really expect this to change.
   s_pageSize = Heap.pageSize();
   
   // create the page mask, which is all bit off past the page size.
   s_pageMask = 0;
   for ( uint32 bit = 0; bit < sizeof( int32 ); bit++ )
   {
      int32 mask = 1 << bit;
      
      if( mask >= s_pageSize )
         s_pageMask |= mask;
   }
}


SmallMemBlockAlloc::PAGE_HEADER* SmallMemBlockAlloc::newPage( int blockSize )
{
   PAGE_HEADER* p = (PAGE_HEADER*) Heap.allocPage();
   p->next = 0;
   p->prev = 0;
   p->firstFree = p + (s_pageSize - blockSize);
   p->allocated = 0;
   
   if ( blockSize <= 8 )
   {
      p->pageArea = 0;
   }
   else if ( blockSize <= 16 )
   {
      p->pageArea = 1;
   }
   else if ( blockSize <= 32 )
   {
      p->pageArea = 2;
   }
   else if( blockSize <= 64 )
   {
      p->pageArea = 3;
   }
   else
      fassert( false );
   
   return p;
}

SmallMemBlockAlloc::~SmallMemBlockAlloc()
{
   delete m_mtx;
   
   for( int i = 0; i < page_list_size; i ++ )
   {
      PAGE_HEADER* p = page_lists[i];
      
      while( p != 0 )
      {
         PAGE_HEADER* next = p->next;
         Heap.freePage( p );
         p = next;
      }
   }
   
}
   
void* SmallMemBlockAlloc::alloc( unsigned int bytes )
{
   register int index;
   
   if ( bytes <= 8 )
   {
      index = 0;
   }
   else if ( bytes <= 16 )
   {
      index = 1;
   }
   else if ( bytes <= 32 )
   {
      index = 2;
   }
   else if( bytes <= 64 )
   {
      index = 3;
   }
   else {
      return 0;
   }

   // got a free node in the list?
   m_mtx->lock();
   
   void* freeNode = page_free_lists[index];
   
   if( freeNode != 0 )
   {
      // cool, allocate it
      PAGE_HEADER* pageNode = (PAGE_HEADER*) ( ((int)freeNode) & s_pageMask );
      // advance our record 
      page_free_lists[index] = *( (void**) freeNode );
      // and account the page
      pageNode->allocated++;

      m_mtx->unlock();
   }
   else 
   {
      register int size = (8 << index);
      
      // see if we have some space on the last page.
      PAGE_HEADER* pageNode = page_lists[index];
      
      if ( ((uint32)pageNode->firstFree - (uint32)pageNode) > sizeof( PAGE_HEADER ) + size  )
      {
         // yay, we got some space
         freeNode = pageNode->firstFree;
         // go back on the page where next free data is located.
         pageNode->firstFree = (void*) ( ((int)pageNode->firstFree) - size );
         // account
         pageNode->allocated++;
         m_mtx->unlock();
      }
      else 
      {
         // no luck, we need a new page.
         // We're doing a lot of system job here. Free the mutex.
         m_mtx->unlock();
      
         pageNode = newPage( size );
         
         freeNode = pageNode->firstFree;
         
         // go back on the page where next free data is located.
         pageNode->firstFree = (void*) ( ((int)pageNode->firstFree) - size );
         
         // account
         pageNode->allocated++;
         
         // Reachieve the lock to store the page we just bought. 
         // It doesn't matter if some space has become available by now,
         // we just got some extra stuff to be used in future. 
         // But we can't relay on cached values.
         m_mtx->lock();
         
         pageNode->prev = page_lists[index];
         page_lists[index]->next = pageNode;
         page_lists[index] = pageNode;
         
         m_mtx->unlock();
      }
   }
   
   // we got a valid node and the mutex unlocked here.
   return freeNode;
}

void SmallMemBlockAlloc::free( void* bytes )
{
   // determine the pagine from which this block
   // is coming from.
   PAGE_HEADER* pageNode = (PAGE_HEADER*) ( ((int)bytes) & s_pageMask );
   
   bool bRelease = false;
   // account for the data being free
   m_mtx->lock();

   if( --pageNode->allocated == 0 )
   {
      // disengage the page from the list, unless it's the only one.
      if( pageNode->next != 0 )
      {
         // we'll release this page when we're out from the mutex
         bRelease = true;
         pageNode->next->prev = pageNode->prev;
         
         // can't be top page if next != 0
      }
      
      if ( pageNode->prev != 0 )
      {
         // we'll release this page when we're out from the mutex
         bRelease = true;
         
         if ( pageNode->next == 0 )
         {
            // the first page. We must update our pointer
            pageNode->prev->next = 0;
            page_lists[ pageNode->pageArea ] = pageNode->prev;
         }
         else 
         {
            // just update the previous node
            pageNode->prev->next = pageNode->next;
         }
      }
   }
   
   m_mtx->unlock();

   // eventually free the page
   if( bRelease )
      Heap.freePage( pageNode );
}

#endif
}

/* end of smba.h */
