/*
   FALCON - The Falcon Programming Language.
   FILE: dict.cpp
   $Id: lineardict.cpp,v 1.11 2007/08/11 00:11:54 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 23 21:55:38 CEST 2004

   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/


#include <falcon/lineardict.h>
#include <falcon/item.h>
#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/vm.h>

#if defined(__BORLANDC__)
   #include <string.h>
#else
   #include <cstring>
#endif

namespace Falcon
{

//=======================================================
// Iterator
//


LinearDictIterator::LinearDictIterator( LinearDict *owner, uint32 pos ):
   m_dictPos( pos ),
   m_dict( owner )
{
   m_versionNumber = owner->version();
}

bool LinearDictIterator::next()
{
   if( m_dict == 0 )
      return false;

   register uint32 size = m_dict->length();



   if ( m_versionNumber == m_dict->version() && size != 0 && size -1 > m_dictPos )
   {
      m_dictPos++;
      return true;
   }

   // also invalidate
   m_dict = 0;
   return false;
}

bool LinearDictIterator::prev()
{
   if( m_dict == 0 )
      return false;

   if ( m_versionNumber == m_dict->version() && m_dictPos > 0 )
   {
      m_dictPos--;
      return true;
   }

   // also invalidate
   m_dict = 0;
   return false;
}


bool LinearDictIterator::isValid() const
{
   return ( m_dict != 0 && m_dict->length() > m_dictPos );
}

bool LinearDictIterator::isOwner( void *collection ) const
{
   return m_dict == collection;
}

void LinearDictIterator::invalidate()
{
   m_dict = 0;
}

Item &LinearDictIterator::getCurrent() const
{
   return m_dict->elementAt( m_dictPos )->value();
}

const Item &LinearDictIterator::getCurrentKey() const
{
   return m_dict->elementAt( m_dictPos )->key();
}

bool LinearDictIterator::hasNext() const
{
   if( m_dict == 0 )
      return false;

   register uint32 size = m_dict->length();
   if ( m_versionNumber == m_dict->version() && size > 0 && size -1 > m_dictPos )
   {
      return true;
   }

   return false;
}

bool LinearDictIterator::hasPrev() const
{
   if( m_dict == 0 )
      return false;

   if ( m_versionNumber == m_dict->version() && m_dictPos > 0 )
   {
      return true;
   }

   return false;
}

bool LinearDictIterator::equal( const CoreIterator &other ) const
{
   if ( ! isValid() && ! other.isValid() )
      return true;

   if ( ! isValid() || ! other.isValid() )
      return false;

   if ( other.isOwner( m_dict ) )
   {
      const LinearDictIterator *oti = static_cast< const LinearDictIterator*>( &other );
      return oti->m_dictPos == m_dictPos;
   }

   return false;
}

//=======================================================
// Iterator
//

LinearDict::LinearDict( VMachine *vm ):
   CoreDict( vm, sizeof( *this ) ),
   m_size(0),
   m_alloc(0),
   m_data(0),
   m_version( 0 ),
   m_travPos( 0 )
{}

LinearDict::LinearDict( VMachine *vm, uint32 size ):
   CoreDict( vm, esize( size ) + sizeof( LinearDict ) ),
   m_version( 0 ),
   m_travPos( 0 )
{
   m_data = (LinearDictEntry *) memAlloc( esize( size ) );
   length(0);
   allocated( size );
}

LinearDict::~LinearDict()
{
   if ( m_data != 0 )
      memFree( m_data );
}

uint32 LinearDict::length() const
{
   return m_size;
}

DictIterator *LinearDict::begin()
{
   return new LinearDictIterator( this, 0 );
}

DictIterator *LinearDict::last()
{
   return new LinearDictIterator( this, m_size - 1 );
}


Item *LinearDict::find( const Item &key )
{
   uint32 posHint;

   // Insert supports substitution semantics.
   if ( findInternal( key, posHint ) ) {
      return &m_data[ posHint ].value();
   }

   return 0;
}

DictIterator *LinearDict::findIterator( const Item &key )
{
   uint32 posHint;

   // Insert supports substitution semantics.
   if ( findInternal( key, posHint ) ) {
      return new LinearDictIterator( this, posHint );
   }

   return 0;
}

bool LinearDict::remove( DictIterator *iter )
{
   if( ! iter->isOwner( this ) || ! iter->isValid() )
      return false;

   LinearDictIterator *ldi = static_cast<LinearDictIterator *>(iter);

   removeAt( ldi->m_dictPos );
   // maintain compatibility
   ldi->m_versionNumber = m_version;
   return true;
}

bool LinearDict::remove( const Item &key )
{
  uint32 posHint;

   // Insert supports substitution semantics.
   if ( findInternal( key, posHint ) ) {
      removeAt( posHint );
      return true;
   }

   return false;
}

void LinearDict::insert( const Item &key, const Item &value )
{
   uint32 posHint;

   // Insert supports substitution semantics.
   if ( findInternal( key, posHint ) ) {
      m_data[ posHint ].value( value );
      return;
   }

   // Entry not found, must be added
   addInternal( posHint, key, value );
}

bool LinearDict::equal( const CoreDict &other ) const
{
   if ( &other == this )
      return true;
   return false;
}


void LinearDict::merge( const CoreDict &dict )
{
   const_cast< CoreDict *>( &dict )->traverseBegin();

   Item key, value;
   while( const_cast< CoreDict *>( &dict )->traverseNext( key, value ) )
   {
      insert( key, value );
   }
}



bool LinearDict::addInternal( uint32 pos, const Item &key, const Item &value )
{
   if ( pos > m_size )
      return false;

   // haven't we got enough space?
   if ( m_alloc <= m_size  )
   {
      m_alloc = m_size + flc_DICT_GROWTH;
      LinearDictEntry *mem = (LinearDictEntry *) memAlloc( esize( m_alloc ) );
      if ( pos > 0 )
         memcpy( mem, m_data, esize( pos ) );

      LinearDictEntry *entry = (LinearDictEntry *) (mem + pos);
      entry->key( key );
      entry->value( value );
      if ( pos < m_size )
         memcpy( mem+pos + 1, m_data + pos, esize( m_size - pos ) );

      if ( m_data != 0 )
         memFree( m_data );
      m_data = mem;
      updateAllocSize( esize( m_alloc ) + sizeof( LinearDict ) );
   }
   else {
      if ( pos < m_size )
         memmove( m_data + pos + 1, m_data + pos,  esize( m_size - pos ) );

      LinearDictEntry *entry = (LinearDictEntry *) (m_data + pos);
      entry->key( key );
      entry->value( value );
  }

   length( m_size + 1 );
   m_version++;
   return true;
}


bool LinearDict::removeAt( uint32 pos )
{
   if ( pos >= m_size )
      return false;

   memmove( m_data + pos, m_data + pos + 1, esize( m_size - pos ) );
   length( m_size - 1 );

   // for now, do not reallocate.
   m_version++;

   return true;
}


bool LinearDict::findInternal( const Item &key, uint32 &ret_pos ) const
{
   uint32 lower = 0, higher, point;
   higher = m_size;
   VMachine *vm = origin();

   if ( higher == 0 ) {
      ret_pos = 0;
      return false;
   }
   higher --;

   point = higher / 2;
   LinearDictEntry *current;
   int comparation;

   while ( true )
   {
      // get the table row
      current = m_data + point;

      comparation = vm->compareItems( key, current->key() );

      if ( comparation == 0 )
      {
         ret_pos = point;
         return true;
      }
      else
      {
         if ( lower == higher )  // not found
         {
            break;
         }
         // last try. In pair sized dictionaries, it can be also in the other node
         else if ( lower == higher -1 )
         {
            // key is EVEN less than the lower one
            if ( comparation < 0  )
            {
               ret_pos = lower;
               return false;
            }

            // being integer math, ulPoint is rounded by defect and has
            // already looked at the ulLower position
            point = lower = higher;
            // try again
            continue;
         }

         if ( comparation > 0 )
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
   ret_pos =  comparation > 0 ? higher + 1 : higher;

   return false;
}


CoreDict *LinearDict::clone() const
{
   if ( m_size == 0 )
      return new LinearDict( origin() );

   LinearDict *ret = new LinearDict( origin(), m_size );
   ret->length( m_size );
   memcpy( ret->m_data, m_data, esize( m_size ) );
   return ret;
}

void LinearDict::traverseBegin()
{
   m_travPos = 0;
}

bool LinearDict::traverseNext( Item &key, Item &value )
{
   if( m_travPos >= m_size )
      return false;

   key = m_data[ m_travPos ].key();
   value = m_data[ m_travPos ].value();
   m_travPos++;
   return true;
}

void LinearDict::clear()
{
   memFree( m_data );
   m_data = 0;
   m_alloc = 0;
   m_size = 0;
   updateAllocSize( sizeof( *this ) );
   m_version++;
}

}

/* end of dict.cpp */
