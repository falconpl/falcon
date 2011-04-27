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
#include <falcon/item.h>
#include <falcon/string.h>

#include <string.h>

#define flc_ARRAY_GROWTH 32

namespace Falcon
{

ItemArray::ItemArray():
   m_alloc(0),
   m_size(0),
   m_data(0),
   m_growth( flc_ARRAY_GROWTH )
{}

ItemArray::ItemArray( const ItemArray& other ):
   m_growth( other.m_growth )
{
   if( other.m_size != 0 )
   {
      m_alloc = other.m_size;
      m_size = other.m_size;
      m_data = new Item[other.m_size];
      memcpy( m_data, other.m_data, esize(other.m_size) );

      // duplicate strings
      /*
      for ( uint32 i = 0; i < m_size; ++i )
      {
         Item& item = m_data[i];

         if( item.isString() )
         {
            item = new CoreString( *item.asString() );
         }         
      }
      */
   }
   else
   {
      m_alloc = 0;
      m_size = 0;
      m_data = 0;
   }
}


ItemArray::ItemArray( length_t prealloc ):
   m_growth( prealloc == 0 ? flc_ARRAY_GROWTH  : prealloc )
{
   if( m_growth < 4 )
      m_growth = 4;

   m_data = new Item[m_growth];
   m_alloc = m_growth;
   m_size = 0;
}


ItemArray::ItemArray( Item *buffer, length_t size, length_t alloc ):
   m_alloc(alloc),
   m_size(size),
   m_data(buffer),
   m_growth( flc_ARRAY_GROWTH )
{}


ItemArray::~ItemArray()
{
   delete[] m_data;
}

void ItemArray::append( const Item &ndata )
{
   // create enough space to hold the data
   if ( m_alloc <= m_size )
   {
      m_alloc = m_size + m_growth;

      Item* newData = new Item[m_alloc];
      memcpy( newData, m_data, m_size );

      // ndata may come from m_data; delete it AFTER having assigned it.
      newData[ m_size ] = ndata;

      delete[] m_data;
      m_data = newData;

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

      Item* newData = new Item[m_alloc];
      memcpy( newData, m_data, m_size );
      delete[] m_data;
      m_data = newData;
   }

   memcpy( m_data + m_size, other.m_data, esize( other.m_size ) );
   m_size += other.m_size;
}

void ItemArray::prepend( const Item &ndata )
{
   // create enough space to hold the data
   Item *mem = new Item[m_size + 1];
   m_alloc = m_size + 1;
   if ( m_size != 0 )
      memcpy( mem + 1, m_data, esize( m_size ) );
   mem[0] = ndata;
   if ( m_size != 0 )
      delete[] m_data;
   m_data = mem;
   m_size++;
}

void ItemArray::merge_front( const ItemArray &other )
{
   if ( other.m_size == 0 )
      return;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      Item *mem = new Item [m_alloc];
      memcpy( mem , other.m_data, esize( other.m_size ) );
      if ( m_size > 0 ) {
         memcpy( mem + other.m_size, m_data, esize( m_size ) );
         delete[] m_data;
      }

      m_size = m_alloc;
      m_data = mem;
   }
   else {
      memmove( m_data + other.m_size, m_data, esize( m_size ) );
      memcpy( m_data, other.m_data, esize( other.m_size ) );
      m_size += other.m_size;
   }
   
}

bool ItemArray::insert( const Item &ndata, length_t pos )
{
   if ( pos > m_size )
      return false;

   if ( m_alloc <= m_size ) {
      m_alloc = m_size + flc_ARRAY_GROWTH;
      Item *mem = new Item[m_alloc];
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );
      if ( pos < (int32)m_size )
         memcpy( mem + pos + 1, m_data + pos , esize(m_size - pos) );

      mem[ pos ] = ndata;
      m_size++;
      delete[] m_data;
      m_data = mem;
   }
   else {
      if ( pos < (int32)m_size )
         memmove( m_data + pos + 1, m_data+pos, esize( m_size - pos) );
      m_data[pos] = ndata;
      m_size ++;
   }
      
   return true;
}

bool ItemArray::insert( const ItemArray &other, length_t pos )
{
   if ( other.m_size == 0 )
      return true;

   if ( pos > m_size )
      return false;

   if ( m_alloc < m_size + other.m_size ) {
      m_alloc = m_size + other.m_size;
      Item *mem = new Item[m_alloc];
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );

      if ( pos < (int32)m_size )
         memcpy( mem + pos + other.m_size, m_data + pos , esize(m_size - pos) );

      memcpy( mem + pos , other.m_data, esize( other.m_size ) );

      m_size = m_alloc;
      delete[] m_data;
      m_data = mem;
   }
   else {
      if ( pos < (int32)m_size )
         memmove( m_data + other.m_size + pos, m_data + pos, esize(m_size - pos ) );
      memcpy( m_data + pos , other.m_data, esize( other.m_size ) );
      m_size += other.m_size;
   }

   return true;
}

bool ItemArray::remove( length_t first, length_t rsize )
{
   if (rsize == 0 )
   {
      return true;
   }
   
   if ( first + rsize > m_size )
      return false;

   length_t last = first + rsize;
   if ( last < m_size )
      memmove( m_data + first, m_data + last, esize(m_size - last ) );
   m_size -= rsize;
      
   return true;
}

bool ItemArray::remove( length_t first )
{
   if ( first >= m_size )
      return false;

   if ( first < (int32)m_size - 1 )
      memmove( m_data + first, m_data + first + 1, esize(m_size - first) );
   m_size --;

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

bool ItemArray::change( const ItemArray &other, length_t begin, length_t rsize )
{
   length_t end = begin + rsize;

   if( end > m_size )
      return false;

   // we're considering end as "included" from now on.
   // this considers also negative range which already includes their extreme.
   if ( m_size - rsize + other.m_size > m_alloc )
   {
      m_alloc =  m_size - rsize + other.m_size;
      Item *mem = new Item[m_alloc];
      if ( begin > 0 )
         memcpy( mem, m_data, esize( begin ) );
      if ( other.m_size > 0 )
         memcpy( mem + begin, other.m_data, esize( other.m_size ) );

      if ( end < (int32) m_size )
         memcpy( mem + begin + other.m_size, m_data + end, esize(m_size - end) );

      delete[] m_data;
      m_data = mem;
      m_size = m_alloc;
   }
   else {
      if ( end < m_size )
         memmove( m_data + begin + other.m_size, m_data + end, esize(m_size - end) );

      if ( other.m_size > 0 )
         memcpy( m_data + begin, other.m_data, esize( other.m_size ) );
      m_size = m_size - rsize + other.m_size;
   }
   
   return true;
}

bool ItemArray::insertSpace( length_t pos, length_t size )
{
   if ( size == 0 )
      return true;

   if ( pos < 0 )
      pos = m_size + pos;
   if ( pos < 0 || pos > m_size )
      return false;

   if ( m_alloc < m_size + size ) {
      m_alloc = ((m_size + size)/m_growth+1)*m_growth;
      Item *mem = new Item[ m_alloc ];
      if ( pos > 0 )
         memcpy( mem , m_data, esize( pos ) );

      if ( pos < m_size )
         memcpy( mem + pos + size, m_data + pos , esize(m_size - pos) );

      for( length_t i = pos; i < pos + size; i ++ )
         m_data[i] = Item();
      delete[] m_data;
      m_data = mem;

      m_size += size;
   }
   else {
      if ( pos < m_size )
         memmove( m_data + size + pos, m_data + pos, esize(m_size - pos) );
      for( length_t i = pos; i < pos + size; i ++ )
         m_data[i] = Item();
      
      m_size += size;
   }
   
   return true;
}


ItemArray *ItemArray::partition( length_t start, length_t size, bool bReverse ) const
{
   Item *buffer;

   if ( size == 0 )
   {
      return new ItemArray;
   }

   if( bReverse ) {
      buffer = new Item [size];

      for( length_t i = 0; i < size; i ++ )
         buffer[i] = m_data[start - i];
   }
   else
   {
      buffer = new Item[size];
      memcpy( buffer, m_data + start, esize( size )  );
   }

   return new ItemArray( buffer, size, size );
}


void ItemArray::resize( length_t size )
{
   // use this request also to force size in shape with alloc.
   if ( size > m_alloc )
   {
      m_alloc = (size/m_growth + 1) *m_growth;
      Item* newData = new Item[m_alloc];
      memcpy( newData, m_data, esize( m_size ) );
      memset( newData + m_size, 0, esize( m_alloc - m_size ) );

      delete[] m_data;
      m_data = newData;
   }
   else if ( size > m_size )
   {
      memset( m_data + m_size, 0, esize( size - m_size ) );
   }
   else if ( size == m_size )
   {
      return;
   }

   m_size = size;
}

void ItemArray::compact()
{
   if ( m_size == 0 ) {
      if ( m_data != 0 ) {
         delete[] m_data;
         m_data = 0;
      }
      m_alloc = 0;
   }
   else if ( m_size < m_alloc )
   {
      m_alloc = m_size;
      Item* newData = new Item[m_alloc];
      memcpy( newData, m_data, esize( m_size ) );
      memset( newData + m_size, 0, esize( m_alloc - m_size ) );

      delete[] m_data;
      m_data = newData;
   }
}

void ItemArray::reserve( length_t size )
{
   if ( size == 0 )
   {
      if ( m_data != 0 ) {
         delete[] m_data;
         m_data = 0;
      }
      m_alloc = 0;
      m_size = 0;
   }
   else if ( size > m_alloc ) 
   {
      Item* newData = new Item[m_alloc];
      memcpy( newData, m_data, esize( m_size ) );

      delete[] m_data;
      m_data = newData;

      m_alloc = size;
   }
}

bool ItemArray::copyOnto( length_t from, const ItemArray& src, length_t first, length_t amount )
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

   return true;
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
      /*
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
      */
   }
   
   if( m_size < other.m_size )
      return -1;
   
   //  ok, we're the same
   return 0;
}

}

/* end of itemarray.cpp */
