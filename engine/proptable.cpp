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
#include <falcon/fassert.h>
#include <falcon/error.h>
#include <falcon/vm.h>
#include <falcon/corefunc.h>

#include <string.h>

namespace Falcon
{

PropertyTable::PropertyTable( uint32 size ):
   m_bReflective( false ),
   m_bStatic( true )
{
   m_size = size;
   m_added = 0;
   m_entries = (PropEntry *) memAlloc( sizeof( PropEntry ) * size );
   // Luckily, 0 is the neuter value for all our representations.
   memset( m_entries, 0, sizeof( PropEntry ) * size );
}

PropertyTable::PropertyTable( const PropertyTable &other ):
   m_bReflective( other.m_bReflective ),
   m_bStatic( other.m_bStatic )
{
   m_size = other.m_size;
   m_added = other.m_added;
   m_entries = (PropEntry *) memAlloc( sizeof( PropEntry ) * m_size );
   memcpy( m_entries, other.m_entries, sizeof( PropEntry ) * m_size );
}


PropertyTable::~PropertyTable()
{
   memFree( m_entries );
}


void PropertyTable::checkProperties()
{
   m_bStatic = true;
   m_bReflective = false;

   for( uint32 i = 0; i < m_added; i++ )
   {
      const PropEntry &e = m_entries[i];

      if ( e.m_eReflectMode != e_reflectNone ) {
         m_bReflective = true;
      }
      else if ( ! e.m_bReadOnly ) {
         m_bStatic = false;
      }
   }
}


bool PropertyTable::findKey( const String &key, uint32 &pos ) const
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
      current = m_entries[point].m_name;

      if( *current == key ) {
         pos = point;
         return true;
      }
      else
      {
         if ( lower == higher -1 )
         {
            // key is EVEN less than the lower one
            if ( key < *current )
            {
               pos = lower;
               return false;
            }

            // being integer math, ulPoint is rounded by defect and has
            // already looked at the ulLower position
            if ( key == *m_entries[higher].m_name ) {
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

         if ( key > *current )
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
   pos =  key > *current ? higher + 1 : higher;

   return false;
}


PropEntry &PropertyTable::append( const String *key )
{
   if( m_added >= m_size ) return m_entries[0];

   if ( m_added != 0 )
   {
      uint32 pos;
      if ( !findKey( *key, pos ) )
      {
         memmove( m_entries + pos, m_entries + pos + 1, sizeof( PropEntry ) * (m_added - pos) );
         m_entries[pos].m_name = key;
      }

      m_added++;
      return m_entries[pos];
   }
   else
   {
      return appendSafe( key );
   }
}


void PropEntry::reflectTo( CoreObject *instance, void *user_data, const Item &prop ) const
{
   byte *ud = (byte *) user_data;

   switch( m_eReflectMode )
   {
      case e_reflectBool:
         ((bool *) user_data)[ m_reflection.offset ] = prop.isTrue();
         break;

      case e_reflectByte:
         ud[ m_reflection.offset ] = (byte) prop.forceInteger();
         break;

      case e_reflectChar:
         ((char *) user_data)[ m_reflection.offset ] = (char) prop.forceInteger();
         break;

      case e_reflectShort:
         *(int16 *) (ud + m_reflection.offset) = (int16) prop.forceInteger();
         break;

      case e_reflectUShort:
         *(uint16 *)(ud + m_reflection.offset) = (uint16) prop.forceInteger();
         break;

      case e_reflectInt:
         *(int32 *) (ud + m_reflection.offset) = (int32) prop.forceInteger();
         break;

      case e_reflectUInt:
         *(uint32 *)(ud + m_reflection.offset) = (uint32) prop.forceInteger();
         break;

      case e_reflectLong:
         *(long *)(ud + m_reflection.offset) = (long) prop.forceInteger();
         break;

      case e_reflectULong:
         *(unsigned long *)(ud + m_reflection.offset) = (unsigned long) prop.forceInteger();
         break;

      case e_reflectLL:
         *(int64 *)(ud + m_reflection.offset) =  prop.forceInteger();
         break;

      case e_reflectULL:
         *(uint64 *)(ud + m_reflection.offset) =  (uint64) prop.forceInteger();
         break;

      case e_reflectFloat:
         *(float *)(ud + m_reflection.offset) =  (float) prop.forceNumeric();
         break;

      case e_reflectDouble:
         *(double *)(ud + m_reflection.offset) =  (double) prop.forceNumeric();
         break;

      case e_reflectFunc:
         // We should not have been called if "to" was zero; we're read only.
         fassert( m_reflection.rfunc.to != 0 );
         m_reflection.rfunc.to( instance, user_data, *const_cast<Item *>(&prop), *this );
         break;

      case e_reflectSetGet:
         // reflection is via a private setter function.
         // We should not have been called if "to" was NO_OFFSET; we're read only.
		  fassert( m_reflection.gs.m_setterId != PropEntry::NO_OFFSET );
         {
            VMachine* vm = VMachine::getCurrent();
            const Item* call = instance->generator()->properties()
                        .getValue( m_reflection.gs.m_setterId );

            fassert( call->isCallable() );
            if( vm->currentSymbol() == call->asFunction()->symbol() )
            {
               // can't be
               throw new AccessError( ErrorParam( e_prop_loop, __LINE__ )
                     .origin( e_orig_vm )
                     .extra( *m_name ));
            }

            Item method = *call;
            method.methodize( instance );

            vm->pushParam( prop );
            VMachine::getCurrent()->callItemAtomic( method, 1 );
         }
         break;

      default:
         fassert( false );
         break;
   }
}

void PropEntry::reflectFrom( CoreObject *instance, void *user_data, Item &prop ) const
{
   byte *ud = (byte *) user_data;

   switch( m_eReflectMode )
   {
      case e_reflectBool:
         prop.setBoolean( ((bool *) user_data)[ m_reflection.offset ] );
         break;

      case e_reflectByte:
         prop = (int64) ud[ m_reflection.offset ];
         break;

      case e_reflectChar:
         prop = (int64) ((char *) user_data)[ m_reflection.offset ];
         break;

      case e_reflectShort:
         prop = (int64) *(int16 *)(ud + m_reflection.offset);
         break;

      case e_reflectUShort:
         prop = (int64) *(uint16 *)(ud + m_reflection.offset);
         break;

      case e_reflectInt:
         prop = (int64) *(int32 *)(ud + m_reflection.offset);
         break;

      case e_reflectUInt:
         prop = (int64) *(uint32 *)(ud + m_reflection.offset);
         break;

      case e_reflectLong:
         prop = (int64) *(long *)(ud + m_reflection.offset);
         break;

      case e_reflectULong:
         prop = (int64) *(unsigned long *)(ud + m_reflection.offset);
         break;

      case e_reflectLL:
         prop = *(int64 *)(ud + m_reflection.offset);
         break;

      case e_reflectULL:
         prop = (int64) *(uint64 *)(ud + m_reflection.offset);
         break;

      case e_reflectFloat:
         prop = (numeric) *(float *)(ud + m_reflection.offset);
         break;

      case e_reflectDouble:
         prop = (numeric) *(double *)(ud + m_reflection.offset);
         break;

      case e_reflectFunc:
         m_reflection.rfunc.from( instance, user_data, prop, *this );
         break;

      case e_reflectSetGet:
         // reflection is via a private setter function.
         // Write only properties are allowed.
         if(  m_reflection.gs.m_getterId == PropEntry::NO_OFFSET )
         {
            throw new AccessError( ErrorParam( e_prop_wo, __LINE__ )
                  .origin( e_orig_vm )
                  .extra( *m_name ) );
         }
         else
         {
            VMachine* vm = VMachine::getCurrent();
            const Item* call = instance->generator()->properties()
                        .getValue( m_reflection.gs.m_getterId );

            if( vm->currentSymbol() == call->asFunction()->symbol() )
            {
               // can't be
               throw new AccessError( ErrorParam( e_prop_loop, __LINE__ )
                     .origin( e_orig_vm )
                     .extra( *m_name ));
            }

            Item method = *call;
            method.methodize( instance );
            VMachine::getCurrent()->callItemAtomic( method, 0 );
            prop = vm->regA();
         }
         break;
         
      default:
         fassert( false );
         break;
   }
}

}

/* end of proptable.cpp */
