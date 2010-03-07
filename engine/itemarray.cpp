/*
   FALCON - The Falcon Programming Language.
   FILE: itemarray.cpp

   Basic item array structure (sequence).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 27 Jul 2009 20:48:24 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of core arrays.
   Core arrays are meant to hold item pointers.
*/

#include <falcon/itemarray.h>
#include <falcon/iterator.h>
#include <falcon/memory.h>
#include <falcon/vm.h>
#include <falcon/lineardict.h>
#include <falcon/coretable.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>

#include <string.h>


namespace Falcon
{

ItemArray::ItemArray():
   m_alloc(0),
   m_size(0),
   m_data(0),
   m_growth( flc_ARRAY_GROWTH ),
   m_owner(0)
{}

ItemArray::ItemArray( const ItemArray& other ):
   m_growth( other.m_growth ),
   m_owner( 0 )
{
   if( other.m_size != 0 )
   {
      m_alloc = other.m_size;
      m_size = other.m_size;
      m_data = (Item *) memAlloc( esize(other.m_size) );
      memcpy( m_data, other.m_data, esize(other.m_size) );

      // duplicate strings
      for ( uint32 i = 0; i < m_size; ++i )
      {
         Item& item = m_data[i];

         if( item.isString() && item.asString()->isCore() )
         {
            item = new CoreString( *item.asString() );
         }
      }
   }
   else
   {
      m_alloc = 0;
      m_size = 0;
      m_data = 0;
   }
}


ItemArray::ItemArray( uint32 prealloc ):
   m_growth( prealloc == 0 ? flc_ARRAY_GROWTH  : prealloc ),
   m_owner( 0 )
{
   if( m_growth < 4 )
      m_growth = 4;

   m_data = (Item *) memAlloc( esize(m_growth) );
   m_alloc = m_growth;
   m_size = 0;
}


ItemArray::ItemArray( Item *buffer, uint32 size, uint32 alloc ):
   m_alloc(alloc),
   m_size(size),
   m_data(buffer),
   m_growth( flc_ARRAY_GROWTH ),
   m_owner( 0 )
{}


ItemArray::~ItemArray()
{
   if ( m_data != 0 )
      memFree( m_data );
}


void ItemArray::gcMark( uint32 mark )
{
   Sequence::gcMark( mark );

   for( uint32 pos = 0; pos < m_size; pos++ ) {
      memPool->markItem( m_data[pos] );
   }
}


void ItemArray::append( const Item &ndata )
{
   // create enough space to hold the data
   if ( m_alloc <= m_size )
   {
      // we don't know where the item is coming from; it may come from also from our thing.
      Item copy = ndata;
      m_alloc = m_size + m_growth;
      m_data = (Item *) memRealloc( m_data, esize( m_alloc ) );
      m_data[ m_size ] = copy;
   }
   else
   {
      m_data[ m_size ] = ndata;
   }
   m_size++;
}


void ItemArray::merge( const ItemArray &other )
{
   if ( other.m_size == 0 )
      return;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      m_data = (Item *) memRealloc( m_data, esize( m_alloc ) );
   }

   memcpy( m_data + m_size, other.m_data, esize( other.m_size ) );
   m_size += other.m_size;
}

void ItemArray::prepend( const Item &ndata )
{
   // create enough space to hold the data
   Item *mem = (Item *) memAlloc( esize(m_size + 1) );
   m_alloc = m_size + 1;
   if ( m_size != 0 )
      memcpy( mem + 1, m_data, esize( m_size ) );
   mem[0] = ndata;
   if ( m_size != 0 )
      memFree( m_data );
   m_data = mem;
   m_size++;
   invalidateAllIters();
}

void ItemArray::merge_front( const ItemArray &other )
{
   if ( other.m_size == 0 )
      return;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      memcpy( mem , other.m_data, esize( other.m_size ) );
      if ( m_size > 0 ) {
         memcpy( mem + other.m_size, m_data, esize( m_size ) );
         memFree( m_data );
      }

      m_size = m_alloc;
      m_data = mem;
   }
   else {
      memmove( m_data + other.m_size, m_data, esize( m_size ) );
      memcpy( m_data, other.m_data, esize( other.m_size ) );
      m_size += other.m_size;
   }
   
   invalidateAllIters();
}

bool ItemArray::insert( const Item &ndata, int32 pos )
{
   if ( pos < 0 )
      pos = m_size + pos;
   if ( pos < 0 || pos > (int32) m_size )
      return false;

   if ( m_alloc <= m_size ) {
      m_alloc = m_size + flc_ARRAY_GROWTH;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );
      if ( pos < (int32)m_size )
         memcpy( mem + pos + 1, m_data + pos , esize(m_size - pos) );

      mem[ pos ] = ndata;
      m_size++;
      memFree( m_data );
      m_data = mem;
   }
   else {
      if ( pos < (int32)m_size )
         memmove( m_data + pos + 1, m_data+pos, esize( m_size - pos) );
      m_data[pos] = ndata;
      m_size ++;
   }
   
   if ( m_iterList != 0 )
   {
      m_invalidPoint = pos;
      invalidateIteratorOnCriterion();
   }
   
   return true;
}

bool ItemArray::insert( const ItemArray &other, int32 pos )
{
   if ( other.m_size == 0 )
      return true;

   if ( pos < 0 )
      pos = m_size + pos;
   if ( pos < 0 || pos > (int32)m_size )
      return false;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );

      if ( pos < (int32)m_size )
         memcpy( mem + pos + other.m_size, m_data + pos , esize(m_size - pos) );

      memcpy( mem + pos , other.m_data, esize( other.m_size ) );

      m_size = m_alloc;
      memFree( m_data );
      m_data = mem;
   }
   else {
      if ( pos < (int32)m_size )
         memmove( m_data + other.m_size + pos, m_data + pos, esize(m_size - pos ) );
      memcpy( m_data + pos , other.m_data, esize( other.m_size ) );
      m_size += other.m_size;
   }

   if ( m_iterList != 0 )
   {
      m_invalidPoint = pos;
      invalidateIteratorOnCriterion();
   }
   
   return true;
}

bool ItemArray::remove( int32 first, int32 last )
{
   if ( first < 0 )
      first = m_size + first;
   if ( first < 0 || first >= (int32)m_size )
      return false;

   if ( last < 0 )
      last = m_size + last;
   if ( last < 0 || last > (int32)m_size )
      return false;

   if( first > last ) {
      int32 temp = first;
      first = last;
      // last can't be < first if it was == size.
      last = temp+1;
   }

   uint32 rsize = last - first;
   if ( last < (int32)m_size )
      memmove( m_data + first, m_data + last, esize(m_size - last) );
   m_size -= rsize;
   
   if ( m_iterList != 0 )
   {
      m_invalidPoint = first;
      invalidateIteratorOnCriterion();
   }
   
   return true;
}

int32 ItemArray::find( const Item &itm ) const
{
   for( uint32 i = 0; i < m_size; i ++ )
   {
      if ( itm == m_data[ i ] )
         return (int32) i;
   }

   return -1;
}


bool ItemArray::remove( int32 first )
{
   if ( first < 0 )
      first = m_size + first;
   if ( first < 0 || first >= (int32)m_size )
      return false;

   if ( first < (int32)m_size - 1 )
      memmove( m_data + first, m_data + first + 1, esize(m_size - first) );
   m_size --;
   
   if ( m_iterList != 0 )
   {
      m_invalidPoint = first;
      invalidateIteratorOnCriterion();
   }
   
   return true;
}

bool ItemArray::change( const ItemArray &other, int32 begin, int32 end )
{
   if ( begin < 0 )
      begin = m_size + begin;
   if ( begin < 0 || begin > (int32)m_size )
      return false;

   if ( end < 0 )
      end = m_size + end;
   if ( end < 0 || end > (int32)m_size )
      return false;

   if( begin > end ) {
      int32 temp = begin;
      begin = end;
      end = temp+1;
   }

   int32 rsize = end - begin;

   // we're considering end as "included" from now on.
   // this considers also negative range which already includes their extreme.
   if ( m_size - rsize + other.m_size > m_alloc )
   {
      m_alloc =  m_size - rsize + other.m_size;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( begin > 0 )
         memcpy( mem, m_data, esize( begin ) );
      if ( other.m_size > 0 )
         memcpy( mem + begin, other.m_data, esize( other.m_size ) );
      if ( end < (int32) m_size )
         memcpy( mem + begin + other.m_size, m_data + end, esize(m_size - end) );

      memFree( m_data );
      m_data = mem;
      m_size = m_alloc;
   }
   else {
      if ( end < (int32)m_size )
         memmove( m_data + begin + other.m_size, m_data + end, esize(m_size - end) );

      if ( other.m_size > 0 )
         memcpy( m_data + begin, other.m_data, esize( other.m_size ) );
      m_size = m_size - rsize + other.m_size;
   }
   
   if ( m_iterList != 0 )
   {
      m_invalidPoint = begin;
      invalidateIteratorOnCriterion();
   }

   return true;
}

bool ItemArray::insertSpace( uint32 pos, uint32 size )
{
   if ( size == 0 )
      return true;

   if ( pos < 0 )
      pos = m_size + pos;
   if ( pos < 0 || pos > m_size )
      return false;

   if ( m_alloc < m_size + size ) {
      m_alloc = ((m_size + size)/m_growth+1)*m_growth;
      Item *mem = (Item *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );

      if ( pos < m_size )
         memcpy( mem + pos + size, m_data + pos , esize(m_size - pos) );

      for( uint32 i = pos; i < pos + size; i ++ )
         m_data[i] = Item();

      m_size += size;
      memFree( m_data );
      m_data = mem;
   }
   else {
      if ( pos < m_size )
      memmove( m_data + size + pos, m_data + pos, esize(m_size - pos) );
      for( uint32 i = pos; i < pos + size; i ++ )
         m_data[i] = Item();
      m_size += size;
   }
   
   if ( m_iterList != 0 )
   {
      m_invalidPoint = pos;
      invalidateIteratorOnCriterion();
   }

   return true;
}


ItemArray *ItemArray::partition( int32 start, int32 end ) const
{
   int32 size;
   Item *buffer;

   if ( start < 0 )
      start = m_size + start;
   if ( start < 0 || start >= (int32)m_size )
      return 0;

   if ( end < 0 )
      end = m_size + end;
   if ( end < 0 || end > (int32)m_size )
      return 0;

   if( end < start ) {
      size = start - end + 1;
      buffer = (Item *) memAlloc( esize( size ) );

      for( int i = 0; i < size; i ++ )
         buffer[i] = m_data[start - i];
   }
   else {
      if( end == start ) {
         return new ItemArray();
      }
      size = end - start;
      buffer = (Item *) memAlloc( esize( size ) );
      memcpy( buffer, m_data + start, esize( size )  );
   }

   return new ItemArray( buffer, size, size );
}


ItemArray *ItemArray::clone() const
{
   return new ItemArray( *this );
}


void ItemArray::resize( uint32 size )
{
   // use this request also to force size in shape with alloc.
   if ( size > m_alloc ) {
      m_alloc = (size/m_growth + 1) *m_growth;
      m_data = (Item *) memRealloc( m_data, esize( m_alloc ) );
      memset( m_data + m_size, 0, esize( m_alloc - m_size ) );
   }
   else if ( size > m_size )
      memset( m_data + m_size, 0, esize( size - m_size ) );
   else if ( size == m_size )
   {
      return;
   }
   else
   {
      if( m_iterList != 0 )
      {
         m_invalidPoint = size;
         invalidateIteratorOnCriterion();
      }
   }

   m_size = size;
}

void ItemArray::compact() {
   if ( m_size == 0 ) {
      if ( m_data != 0 ) {
         memFree( m_data );
         m_data = 0;
      }
      m_alloc = 0;
   }
   else if ( m_size < m_alloc )
   {
      m_alloc = m_size;
      m_data = (Item *) memRealloc( m_data, esize( m_alloc ) );
      memset( m_data + m_size, 0, esize( m_alloc - m_size ) );
   }
}

void ItemArray::reserve( uint32 size )
{
   if ( size == 0 )
   {
      if( m_iterList != 0 )
      {
         m_invalidPoint = size;
         invalidateIteratorOnCriterion();
      }

      if ( m_data != 0 ) {
         memFree( m_data );
         m_data = 0;
      }
      m_alloc = 0;
      m_size = 0;
   }
   else if ( size > m_alloc ) {
      m_data = (Item *) memRealloc( m_data, esize( size ) );
      m_alloc = size;
   }
}

bool ItemArray::copyOnto( uint32 from, const ItemArray& src, uint32 first, uint32 amount )
{
   if( first > src.length() )
      return false;

   if ( first + amount > src.length() )
      amount = src.length() - first;

   if ( from > length() )
      return false;

   if ( from + amount > length() )
      resize( from + amount );

   memcpy( m_data + from, src.m_data + first, esize( amount ) );
   
   if( m_iterList != 0 )
   {
      m_invalidPoint = from;
      invalidateIteratorOnCriterion();
   }
   
   return true;
}

//============================================================
// Iterator management.
//============================================================

void ItemArray::getIterator( Iterator& tgt, bool tail ) const
{
   Sequence::getIterator( tgt, tail );
   tgt.position( tail ? (length()>0? length()-1: 0) : 0 );
}


void ItemArray::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
   tgt.position( source.position() );
}


void ItemArray::insert( Iterator &iter, const Item &data )
{
   if ( ! iter.isValid() )
      throw new CodeError( ErrorParam( e_invalid_iter, __LINE__ )
            .origin( e_orig_runtime ).extra( "ItemArray::insert" ) );

   insert( data, (int32) iter.position() );
   m_invalidPoint = (uint32) iter.position()+1;
   invalidateIteratorOnCriterion();
   // the iterator inserted before this position, so the element has moved forward
   iter.position( iter.position() + 1 ); 
}

void ItemArray::erase( Iterator &iter )
{
   if ( iter.position() >= length() )
      throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
            .origin( e_orig_runtime ).extra( "ItemArray::erase" ) );

   uint32 first = (uint32) iter.position();
   if ( first+1 < m_size )
      memmove( m_data + first, m_data + first + 1, esize(m_size - first) );
   m_size --;
   
   invalidateAnyOtherIter( &iter );   
}


bool ItemArray::hasNext( const Iterator &iter ) const
{
   return iter.position()+1 < length();
}


bool ItemArray::hasPrev( const Iterator &iter ) const
{
   return iter.position() > 0;
}

bool ItemArray::hasCurrent( const Iterator &iter ) const
{
   return iter.position() < length();
}


bool ItemArray::next( Iterator &iter ) const
{
   if ( iter.position() < length())
   {
      iter.position( iter.position() + 1 );
      return iter.position() < length();
   }
   return false;
}


bool ItemArray::prev( Iterator &iter ) const
{
   if ( iter.position() > 0 )
   {
      iter.position( iter.position() - 1 );
      return true;
   }

   iter.position( length() );
   return false;
}

Item& ItemArray::getCurrent( const Iterator &iter )
{
   if ( iter.position() < length() )
      return m_data[ iter.position() ];

   throw new AccessError( ErrorParam( e_iter_outrange, __LINE__ )
         .origin( e_orig_runtime ).extra( "ItemArray::getCurrent" ) );
}


Item& ItemArray::getCurrentKey( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ )
         .origin( e_orig_runtime ).extra( "ItemArray::getCurrentKey" ) );
}


bool ItemArray::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return first.position() == second.position();
}

bool ItemArray::onCriterion( Iterator* elem ) const
{
   return elem->position() >= m_invalidPoint;
}


int ItemArray::compare( const ItemArray& other, ItemArray::Parentship* parent ) const
{
   // really the same.
   if (&other == this)
      return 0;
      
   // use the size + 1 element to store the parent list
   Parentship current( this, parent );
   
   for ( uint32 i = 0; i < m_size; i ++ )
   {
      // is the other shorter?
      if ( i >= other.m_size )
      {
         // we're bigger
         return 1;
      }
         
      // different arrays?
      if ( m_data[i].isArray() && other.m_data[i].isArray() )
      {
         // check if m_data[i] is in the list of parents.
         ItemArray* ia = &m_data[i].asArray()->items();
         Parentship *p1 = parent;
         // If it is not, we should scan it too.
         bool bDescend = true;
         
         while( p1 != 0 )
         {
            if( p1->m_array == ia )
            {
               bDescend = false;
               break;
            }
            p1 = p1->m_parent;
         }
         
         if ( bDescend )
         {
            int cval = ia->compare( other.m_data[i].asArray()->items(), &current );
            // if the things below us aren't equal, we're not equal
            if ( cval != 0 )
               return cval;
            // else, check other items.
         }
      }
      else 
      {
            int cval = m_data[i].compare( other.m_data[i] );
            // if the things below us aren't equal, we're not equal
            if ( cval != 0 )
               return cval;
            // else, check other items.
      }
   }
   
   if( m_size < other.m_size )
      return -1;
   
   //  ok, we're the same
   return 0;
}

}

/* end of itemarray.cpp */
