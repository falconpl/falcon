/*
   FALCON - The Falcon Programming Language.
   FILE: cacheobject.cpp

   Cache Object - Standard instance of classes in script - base
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom dic 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon Object - Standard instance of classes in script - base
*/

#include <falcon/globals.h>
#include <falcon/cacheobject.h>
#include <falcon/string.h>
#include <falcon/stream.h>

#include <falcon/vm.h>

namespace Falcon {

CacheObject::CacheObject( const CoreClass *generator, bool bSeralizing ):
   CoreObject( generator ),
   m_cache( 0 )
{
   const PropertyTable &pt = m_generatedBy->properties();
   m_cache = (Item *) memAlloc( sizeof( Item ) * pt.added() );

   // We don't need to copy the default values if we're going to overwrite them.
   if ( ! bSeralizing )
   {
      // TODO: Check if a plain copy would do.
      for ( uint32 i = 0; i < pt.added(); i ++ )
      {
         const Item &itm = *pt.getValue(i);
         m_cache[i] = itm.isString() ? new CoreString( *itm.asString() ) :
                     itm;
      }
   }
}


CacheObject::CacheObject( const CacheObject &other ):
   CoreObject( other.generator() ),
   m_cache( 0 )
{
   const PropertyTable &pt = m_generatedBy->properties();

   m_cache = (Item *) memAlloc( sizeof( Item ) * pt.added() );
   for( uint32 i = 0; i < pt.added(); i++ )
   {
      const Item &itm = other.m_cache[i];
      //TODO: GARBAGE
      m_cache[i] = itm.isString() ? new CoreString( *itm.asString() ) :
                        itm;
   }
}

CacheObject::~CacheObject()
{
   memFree( m_cache );
}

void CacheObject::gcMark( uint32 mk )
{
   if( mk != mark() )
   {
      CoreObject::gcMark( mk );

      const PropertyTable &props = m_generatedBy->properties();
      for ( uint32 i = 0; i < props.added(); i ++ )
      {
        memPool->markItem( m_cache[i] );
      }
   }
}

bool CacheObject::setProperty( const String &propName, const Item &value )
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   if ( pt.findKey( propName, pos ) )
   {
      if ( value.isReference() )
         m_cache[ pos ] = value;
      else
         *m_cache[ pos ].dereference() = value;
      return true;
   }

   return false;
}


bool CacheObject::getProperty( const String &propName, Item &ret ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();

   if ( pt.findKey( propName, pos ) )
   {
      ret = *m_cache[pos].dereference();
      return true;
   }

   return false;
}

bool CacheObject::serialize( Stream *stream, bool bLive ) const
{
   uint32 len = m_generatedBy->properties().added();
   uint32 elen = endianInt32( m_generatedBy->properties().added() );
   stream->write((byte *) &elen, sizeof(elen) );

   for( uint32 i = 0; i < len; i ++ )
   {
      if ( m_cache[i].serialize( stream, bLive ) != Item::sc_ok )
         return false;
   }

   CoreObject::serialize( stream, bLive );

   return stream->good();
}


bool CacheObject::deserialize( Stream *stream, bool bLive )
{
   VMachine *vm = VMachine::getCurrent();

   // Read the class property table.
   uint32 len;
   stream->read( (byte *) &len, sizeof( len ) );
   len = endianInt32( len );
   if ( len != m_generatedBy->properties().added() )
   {
      return false;
   }

   for( uint32 i = 0; i < len; i ++ )
   {
      if( ! stream->good() ) {
         return false;
      }

      Item temp;
      Item::e_sercode sc = m_cache[i].deserialize( stream, vm );
      if ( sc != Item::sc_ok )
      {
         return false;
      }
   }

   CoreObject::deserialize( stream, bLive );

   return true;
}


void CacheObject::reflectFrom( void *user_data )
{
   const PropertyTable &pt = m_generatedBy->properties();

   for ( uint32 i = 0; i < pt.added(); i ++ )
   {
      const PropEntry &entry = pt.getEntry(i);
      if( entry.m_eReflectMode != e_reflectNone )
      {
         entry.reflectFrom( this, user_data, m_cache[i] );
      }
   }
}


void CacheObject::reflectTo( void *user_data ) const
{
   const PropertyTable &pt = m_generatedBy->properties();

   for ( uint32 i = 0; i < pt.added(); i ++ )
   {
      const PropEntry &entry = pt.getEntry(i);
      if( entry.m_eReflectMode != e_reflectNone )
      {
         entry.reflectTo( const_cast<CacheObject *>(this), user_data, m_cache[i] );
      }
   }
}


}

/* end of cacheobject.cpp */
