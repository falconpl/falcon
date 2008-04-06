/*
   FALCON - The Falcon Programming Language.
   FILE: proptable.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ott 14 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of the property table for live object and classes.
*/

#include <falcon/proptable.h>
#include <falcon/memory.h>
#include <string.h>

namespace Falcon
{

PropertyTable::PropertyTable( uint32 size )
{
   m_size = size;
   m_added = 0;
   m_keys = (const String **) memAlloc( sizeof( String * ) * size );
   m_values = (Item *) memAlloc( sizeof( Item ) * size );
   m_configs = 0;
}

PropertyTable::PropertyTable( const PropertyTable &other )
{
   m_size = other.m_size;
   m_added = other.m_added;
   m_keys = (const String **) memAlloc( sizeof( String * ) * m_size );
   memcpy( m_keys, other.m_keys, m_added * sizeof( String * ) );
   m_values = (Item *) memAlloc( sizeof( Item ) * m_size );
   memcpy( m_values, other.m_values, m_added * sizeof( Item ) );

   if ( other.m_configs != 0 )
   {
      m_configs = (config*) memAlloc( sizeof( config ) * m_size );
      memcpy( m_configs, other.m_configs, m_added * sizeof( config ) );
   }
   else {
      m_configs = 0;
   }
}

PropertyTable::~PropertyTable()
{
   memFree( m_keys );
   memFree( m_values );
   if ( m_configs != 0 )
      memFree( m_configs );
}

bool PropertyTable::findKey( const String *key, uint32 &pos ) const
{
   uint32 lower = 0, higher, point;
   higher = m_size;
   const String *current;

   if ( higher == 0 ) {
      pos = 0;
      return false;
   }
   higher --;

   point = higher / 2;

   while ( true )
   {
      // get the table row
      current = m_keys[point];

      if( *current == *key ) {
         pos = point;
         return true;
      }
      else
      {
         if ( lower == higher -1 )
         {
            // key is EVEN less than the lower one
            if ( *key < *current )
            {
               pos = lower;
               return false;
            }

            // being integer math, ulPoint is rounded by defect and has
            // already looked at the ulLower position
            if ( *key == *m_keys[higher] ) {
               pos = higher;
               return true;
            }

            // we cannot find it
            break;
         }
         else if ( lower == higher )
         {
            // again, can't find it.
            break;
         }

         if ( *key > *current )
         {
            lower = point;
         }
         else
         {
            higher = point;
         }
         point = ( lower + higher ) / 2;
      }
   }

   // entry not found, but signal the best match anyway
   pos =  *key > *current ? higher + 1 : higher;

   return false;
}


bool PropertyTable::append( const String *key, const Item &itm )
{
   if( m_added >= m_size ) return false;

   if ( m_added != 0 ) {
      uint32 pos;
      if ( !findKey( key, pos ) ) {
         memmove( m_keys + pos, m_keys + pos + 1, sizeof( String * ) * (m_added - pos) );
         m_keys[pos] = key;
         memmove( m_values + pos, m_values + pos + 1, sizeof( Item ) * (m_added - pos) );
         if( m_configs != 0 )
         {
            memmove( m_configs + pos, m_configs + pos + 1, sizeof( config ) * (m_added - pos) );
            m_configs[pos].m_offset = 0;
            m_configs[pos].m_size = 0;
         }
      }
      *getValue( pos ) = itm;
   }
   else {
      m_keys[0] = key;
      *getValue( 0 ) = itm;
   }

   m_added++;

   return true;
}

bool PropertyTable::append( const String *key, const Item &itm, const PropertyTable::config &cfg )
{
   if( m_added >= m_size ) return false;

   if ( m_configs== 0 )
   {
      m_configs = (config*) memAlloc( sizeof( config ) * m_size );
      memset( m_configs, 0, m_size * sizeof( config ) );
   }

   if ( m_added != 0 ) {
      uint32 pos;
      if ( !findKey( key, pos ) ) {
         memmove( m_keys + pos, m_keys + pos + 1, sizeof( String * ) * (m_added - pos) );
         m_keys[pos] = key;
         memmove( m_values + pos, m_values + pos + 1, sizeof( Item ) * (m_added - pos) );

         memmove( m_configs + pos, m_configs + pos + 1, sizeof( config ) * (m_added - pos) );
         m_configs[pos] = cfg;
      }
      *getValue( pos ) = itm;
   }
   else {
      m_keys[0] = key;
      *getValue(0) = itm;
      m_configs[0] = cfg;
   }

   m_added++;

   return true;
}

void PropertyTable::appendSafe( const String *key, const Item &itm, const PropertyTable::config &cfg )
{
   if ( m_configs == 0 )
   {
      m_configs = (config*) memAlloc( sizeof( config ) * m_size );
      memset( m_configs, 0, m_size * sizeof( config ) );
   }
   appendSafe( key, itm );
   m_configs[m_added -1] = cfg;
}

}


/* end of proptable.cpp */
