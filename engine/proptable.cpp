/*
   FALCON - The Falcon Programming Language.
   FILE: proptable.cpp
   $Id: proptable.cpp,v 1.5 2007/01/17 17:01:25 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ott 14 2005
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
   m_keys = (const String **) memAlloc( sizeof( char * ) * size );
   m_values = (Item *) memAlloc( sizeof( Item ) * size );
}

PropertyTable::PropertyTable( const PropertyTable &other )
{
   m_size = other.m_size;
   m_added = other.m_added;
   m_keys = (const String **) memAlloc( sizeof( char * ) * m_size );
   memcpy( m_keys, other.m_keys, m_added * sizeof( char * ) );
   m_values = (Item *) memAlloc( sizeof( Item ) * m_size );
   memcpy( m_values, other.m_values, m_added * sizeof( Item ) );
}

PropertyTable::~PropertyTable()
{
   memFree( m_keys );
   memFree( m_values );
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

   if ( m_added == 0 ) {
      uint32 pos;
      if ( !findKey( key, pos ) ) {
         memmove( m_keys + pos, m_keys + pos + 1, sizeof( char * ) * (m_added - pos) );
         m_keys[pos] = key;
         memmove( m_values + pos, m_values + pos + 1, sizeof( Item ) * (m_added - pos) );
      }
      *getValue( pos ) = itm;
   }
   else
      *getValue( 0 ) = itm;

   m_added++;

   return true;
}

}


/* end of proptable.cpp */
