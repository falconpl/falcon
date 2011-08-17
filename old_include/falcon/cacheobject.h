/*
   FALCON - The Falcon Programming Language.
   FILE: cacheobject.h

   Falcon Object - Standard instance of classes in script
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Jan 2009 13:48:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon Object - Standard instance of classes in script
*/

#ifndef FLC_CACHE_OBJECT_H
#define FLC_CACHE_OBJECT_H

#include <falcon/setup.h>
#include <falcon/coreobject.h>
#include <falcon/cclass.h>

namespace Falcon
{

class VMachine;

class FALCON_DYN_CLASS CacheObject: public CoreObject
{
protected:
   mutable Item *m_cache;

   CacheObject( const CoreClass* generator, bool bSeralizing = false );
   CacheObject( const CacheObject &other );

public:

   virtual ~CacheObject();

   virtual bool serialize( Stream *stream, bool bLive ) const;
   virtual bool deserialize( Stream *stream, bool bLive );
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &key, Item &ret ) const;

   /** Mark the items in the cache and the inner data, if it's garbageable.
      If this object contains an external FalconData instance, it gets marked;
      anyhow, this function marks the cache items.
   */
   virtual void gcMark( uint32 mark );

   Item *cachedProperty( const String &name ) const
   {
      register uint32 pos;
      if ( ! m_generatedBy->properties().findKey( name, pos ) )
         return 0;

      return m_cache + pos;
   }

   Item *cachedPropertyAt( uint32 pos ) const {
      if ( pos > m_generatedBy->properties().added() )
         return 0;
      return m_cache + pos;
   }

   /** Reflect an external data into this object.
      This method uses the reflection informations in the generator class property table
      to load the data stored in the user_data parameter into the local item vector cache.

      As this is an optional operation, the base class implementation does nothing.
   */
   virtual void reflectFrom( void *user_data );

   /** Reflect this object into external data.

      This method uses the reflection informations in the generator class property table
      to store a C/C++ copy of the data stored in the local cache of this object into
      an external data.

      As this is an optional operation, the base class implementation does nothing.
   */
   virtual void reflectTo( void *user_data ) const;

};

}

#endif

/* end of cacheobject.h */
