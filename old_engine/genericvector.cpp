/*
   FALCON - The Falcon Programming Language.
   FILE: genericvector.cpp

   Generic vector - a generic vector of elements
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven oct 27 11:02:00 CEST 2006


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/genericvector.h>
#include <falcon/memory.h>
#include <falcon/fassert.h>
#include <falcon/globals.h>
#include <string.h>

namespace Falcon {

GenericVector::GenericVector( const ElementTraits *traits, uint32 prealloc ):
   m_data( 0 ),
	m_size( 0 ),
	m_allocated( prealloc < alloc_block ? alloc_block : prealloc ),
	m_threshold_size( 0 ),
	m_traits(traits)
{
   fassert( traits !=  0 );

   m_itemSize = traits->memSize();
   if ( m_itemSize % 4 != 0 )
      m_itemSize = (m_itemSize/4 + 1) * 4;

   if ( m_allocated > 0 )
      m_data = (byte *) memAlloc( m_allocated * m_itemSize );
}

void GenericVector::init( const ElementTraits *traits, uint32 prealloc )
{
   fassert( traits !=  0 );

   m_size = 0;
   m_allocated = prealloc < alloc_block ? alloc_block : prealloc;

   m_traits = traits;
   m_itemSize = traits->memSize();
   if ( m_itemSize % 4 != 0 )
      m_itemSize = (m_itemSize/4 + 1) * 4;

   if( m_allocated > 0 )
      m_data = (byte *) memAlloc( m_allocated * m_itemSize );
}

GenericVector::~GenericVector()
{
   if( m_traits->owning() )
   {
      for ( uint32 i = 0; i < m_size; i ++ )
      {
         m_traits->destroy( at( i ) );
      }
   }

   memFree( m_data );
}


void GenericVector::insert( void *data, uint32 pos )
{
   if ( pos > m_size )
   {
      return;
   }

   m_size ++;

   if ( m_size >= m_allocated )
   {
      m_allocated = m_size + alloc_block;
      byte *target_data = (byte *) memRealloc( m_data, m_allocated * m_itemSize );
      if ( target_data != 0 )
      {
         m_data = target_data;
      }
   }
   else {
      if ( pos < m_size )
         memmove( m_data + ( m_itemSize * (pos+1) ), m_data + ( m_itemSize * pos ), ( (m_size - pos) * m_itemSize ) );
   }

   m_traits->copy( m_data + ( m_itemSize * pos ), data );
}

bool GenericVector::remove( uint32 pos )
{
   if ( pos >= m_size )
      return false;


   m_traits->destroy( m_data + ( m_itemSize * pos ) );

   if ( pos < m_size-1 )
   {
      memmove( m_data + ( m_itemSize * pos ),  m_data + ( m_itemSize * (pos+1) ), ( m_itemSize * (m_size - pos ) ) );
   }

   m_size --;
   return true;
}

void GenericVector::set( void *data, uint32 pos )
{
   if ( pos >= m_size )
      return;

   byte *target = m_data + ( m_itemSize * pos );
   m_traits->destroy( target );
   m_traits->copy( target, data );
}

void GenericVector::push( void *data )
{
   m_traits->copy( m_data + ( m_itemSize * m_size ), data );
   m_size ++;

   if ( m_size >= m_allocated )
   {
      m_allocated = m_size + alloc_block;
      byte *target_data = (byte *) memRealloc( m_data, m_allocated * m_itemSize );
      m_data = target_data;
   }
}

void GenericVector::reserve( uint32 s )
{
   if ( m_allocated >= s + 1 )
      return;

   byte *mem = (byte *) memAlloc( (s+1) * m_itemSize );

   if( m_allocated > 0 )
   {
      if ( m_size > 0 )
         memcpy( mem, m_data, m_size * m_itemSize );

      memFree( m_data );
   }

   m_data = mem;
   m_allocated = s+1;
}

void GenericVector::resize( uint32 s )
{
   if ( s == m_size )
      return;

   if( s > m_size )
   {
      if ( s >= m_allocated )
      {
         m_allocated = ((s/alloc_block) + 1) * alloc_block;
         byte *mem = (byte *) memRealloc( m_data, m_allocated * m_itemSize );
         m_data = mem;
         if ( mem == 0 )
         {
            // atm, memRealloc should take care of this.
         }
      }

      for( uint32 i = m_size; i <= s; i ++ ) {
         m_traits->init( at( i ) );
      }
   }
   else {
      if ( m_traits->owning() )
      {
         for( uint32 i = s+1; i < m_size; i ++ ) {
            m_traits->destroy( at( i ) );
         }
      }

      if ( m_threshold_size > 0 )
      {
         if ( s + m_threshold_size < m_size )
         {
            m_allocated = ((s/m_threshold_size) + 1 ) * m_threshold_size;
            m_data = (byte *) memRealloc( m_data, m_allocated * m_itemSize );
            m_allocated = m_allocated;
         }
      }
      else {
         // without threshold size, always resize.
         m_data = (byte *) memRealloc( m_data, (s+1) * m_itemSize );
         m_allocated = s+1;
      }
   }

   m_size = s;
}


}

/* end of genericvector.cpp */
