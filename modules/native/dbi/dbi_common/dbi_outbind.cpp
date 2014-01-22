/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_outbind.cpp

   Database Interface
   Helper for general C-to-Falcon variable binding
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 17 May 2010 22:32:39 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_outbind.h>
#include <falcon/fassert.h>

#include <string.h>
#include <stdlib.h>

namespace Falcon {

//=============================================================
// Output bind single item.
//=============================================================


DBIOutBind::DBIOutBind():
   m_allocated( bufsize ),
   m_allBlockSizes( 0 ),
   m_memory( m_stdBuffer ),
   m_headBlock(0)
{
}


DBIOutBind::~DBIOutBind()
{
   if( m_memory != 0 && m_memory != m_stdBuffer )
   {
      free( m_memory );
      m_memory = 0;
   }

   void* block = m_headBlock;
   while( block != 0 )
   {
      fassert( sizeof(long) == sizeof(void*) );
      long* l = (long*) block;
      l = l-2;
      void *nblock = (void*) l[0];
      free( l );
      block = nblock;
   }

   m_headBlock = m_tailBlock = 0;
}


void* DBIOutBind::allocBlock( unsigned size )
{
   fassert( sizeof(long) == sizeof(void*) );
   long* lblock = (long*) malloc( size + (sizeof(long)*2));
   lblock[0] = 0;
   lblock[1] = (long) size;

   lblock += 2;

   if ( m_tailBlock == 0 )
   {
      fassert( m_headBlock == 0 );
      m_tailBlock = m_headBlock = lblock;
   }
   else
   {
      long* tail = (long*) m_tailBlock;
      tail = tail - 2;
      tail[0] = (long) lblock;
      m_tailBlock = lblock;
   }

   return lblock;
}

void DBIOutBind::setBlockSize( void* block, unsigned size )
{
   long* lblock = (long*) block;
   lblock -= 2;
   m_allBlockSizes += size - lblock[1];
   lblock[1] = size;
}


void* DBIOutBind::consolidate()
{
   if( m_memory != 0 && m_memory != m_stdBuffer )
   {
      free( m_memory );
   }

   if( m_allocated == 0 )
   {
      m_memory = 0;
      return 0;
   }

   m_memory = malloc( m_allocated );
   char* memory = (char*) m_memory;
   m_allocated = 0;

   long* head = (long*) m_headBlock;
   while( head != 0 )
   {
      head -= 2;
      memcpy( memory + m_allocated, head + 2, head[1] );
      m_allocated += head[1];
      long* old = head;
      head = (long*)head[0];
      free( old );
   }

   return m_memory;
}


void* DBIOutBind::alloc( unsigned size )
{
   if( m_memory == 0 || m_memory == m_stdBuffer )
   {
      m_memory  = malloc( size );
   }
   else
   {
      m_memory = realloc( m_memory, size );
   }

   m_allocated = size;
   return m_memory;
}


void* DBIOutBind::reserve( unsigned size )
{
   if( m_headBlock != 0 )
      consolidate();

   if( m_allocated >= size )
      return m_memory;

   if( m_memory == 0 || m_memory == m_stdBuffer )
   {
      m_memory  = malloc( size );
   }
   else
   {
      m_memory = realloc( m_memory, size );
   }

   m_allocated = size;
   return m_memory;
}

void* DBIOutBind::getMemory()
{
   if( m_memory == 0 || m_memory == m_stdBuffer )
   {
      return 0;
   }

   void* mem = m_memory;
   m_allocated = 0;
   m_memory = 0;
   return mem;
}


}

/* end of dbi_outbind.cpp */
