/*
   FALCON - The Falcon Programming Language.
   FILE: cobject.cpp

   Core object implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom dic 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core object implementation.
*/

#include <falcon/vm.h>
#include <falcon/item.h>
#include <falcon/cobject.h>
#include <falcon/symbol.h>
#include <falcon/attribute.h>

namespace Falcon {

CoreObject::CoreObject( const CoreClass *generator,  void *user_data ):
   Garbageable( generator->origin(), sizeof( this ) ),
   m_attributes( 0 ),
   m_generatedBy( generator ),
   m_user_data( user_data ),
   m_cache( 0 )
{
   ObjectManager *om = m_generatedBy->getObjectManager();
   const PropertyTable &pt = m_generatedBy->properties();

   // do we need to create a local cache?
   // if pt is not static and we have not class reflection we need it
   // we need it also if we ask explicitly for it (needCacheData)
   if( om != 0 && ( (! om->hasClassReflection() && ! pt.isStatic()) || om->needCacheData() ) ||
      (! pt.isStatic()) )
   {
      m_cache = (Item *) memAlloc( sizeof( Item ) * pt.added() );
      for ( uint32 i = 0; i < pt.added(); i ++ )
      {
         const Item &itm = *pt.getValue(i);
         m_cache[i] = itm.isString() ? new GarbageString( origin(), *itm.asString() ) :
                      itm;
      }
   }
}

CoreObject::~CoreObject()
{
   if( m_user_data != 0 )
   {
      fassert( m_generatedBy != 0 );

      if ( m_generatedBy->getObjectManager() != 0 )
      {
         m_generatedBy->getObjectManager()->onDestroy( origin(), m_user_data );
      }
   }

   if ( m_cache != 0 )
      memFree( m_cache );

   while( m_attributes != 0 )
   {
      // removing the attribute from this will also cause m_attributes to move forward
      m_attributes->attrib()->removeFrom( this );
   }
}


bool CoreObject::derivedFrom( const String &className ) const
{
   Symbol *clssym = m_generatedBy->symbol();
   return (clssym->name() == className || m_generatedBy->derivedFrom( className ));
}


bool CoreObject::setProperty( const String &propName, const Item &value )
{
   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();

   if ( pt.findKey( propName, pos ) )
   {
      // to be optimized
      setPropertyAt( pos, value );
      return true;
   }

   return false;
}

void CoreObject::setPropertyAt( uint32 pos, const Item &value )
{
   const PropertyTable &pt = m_generatedBy->properties();

   //Ok, we found the property, but what should we do with that?
   const PropEntry &entry = pt.getEntry( pos );
   ObjectManager *mngr = m_generatedBy->getObjectManager();

   // can we write it?
   if ( entry.m_bReadOnly ) {
      origin()->raiseRTError( new AccessError( ErrorParam( e_prop_ro, __LINE__ ) ) );
   }

   if ( entry.m_eReflectMode != e_reflectNone && mngr != 0 && ! mngr->isDeferred() )
   {
      fassert( m_user_data != 0 );

      entry.reflectTo( this, m_user_data, value );
      // remember to cache the value.
   }
   else if ( mngr != 0 && mngr->hasClassReflection() )
   {
      mngr->onSetProperty( this, m_user_data, *entry.m_name, value );
   }

   if ( m_cache != 0 ) {
      if ( value.isReference() )
         m_cache[ pos ] = value;
      else
         *m_cache[ pos ].dereference() = value;
   }
}


bool CoreObject::setProperty( const String &propName, const String &value )
{
   return setProperty( propName, new GarbageString( origin(), value ) );
}


bool CoreObject::getProperty( const String &propName, Item &ret )
{
   register uint32 pos;
   fassert( m_generatedBy != 0 );

   const PropertyTable &pt = m_generatedBy->properties();

   if ( pt.findKey( propName, pos ) )
   {
      // to be optimized.
      getPropertyAt( pos, ret );

      // already assigned, if possible
      return true;
   }

   return false;
}


void CoreObject::getPropertyAt( uint32 pos, Item &ret )
{
   const PropertyTable &pt = m_generatedBy->properties();

   // small debug time security
   fassert( pos < pt.added() );

   //Ok, we found the property, but what should we do with that?
   const PropEntry &entry = pt.getEntry( pos );
   ObjectManager *mngr = m_generatedBy->getObjectManager();

   if ( m_cache != 0 )
   {
      Item &cached = *m_cache[pos].dereference();

      if ( entry.m_eReflectMode != e_reflectNone && mngr != 0 && ! mngr->isDeferred() )
      {
         fassert( m_user_data != 0 );
         // this code allows to modify our cached value.
         entry.reflectFrom( this, m_user_data, cached );
      }
      else if ( mngr != 0 && mngr->hasClassReflection() )
      {
         mngr->onGetProperty( this, m_user_data, *entry.m_name, cached );
      }

      ret = cached;
   }
   else {
      ret = *pt.getValue(pos);

      if ( entry.m_eReflectMode != e_reflectNone && mngr != 0 && ! mngr->isDeferred() )
      {
         fassert( m_user_data != 0 );
         entry.reflectFrom( this, m_user_data, ret );
      }
      else if ( mngr != 0 && mngr->hasClassReflection() )
      {
         mngr->onGetProperty( this, m_user_data, *entry.m_name, ret );
      }
   }

}


CoreObject *CoreObject::clone() const
{
   void *ud = 0;

   // if we can't clone the user data, we should just forbid cloning this object.
   if ( m_user_data != 0 )
   {
      ud = m_generatedBy->getObjectManager()->onClone( origin(), m_user_data );
      if( ud == 0 )
         return 0;
   }

   CoreObject *other = new CoreObject( m_generatedBy, ud );

   // copy attribute list
   AttribHandler *head = m_attributes;
   while( head != 0 )
   {
      head->attrib()->giveTo( other );
      head = head->next();
   }

   return other;
}


bool CoreObject::has( const Attribute *attrib ) const
{
   AttribHandler *head = m_attributes;
   while( head != 0 )
   {
      if ( head->attrib() == attrib )
         return true;

      head =  head->next();
   }

   return false;
}


bool CoreObject::has( const String &attrib ) const
{
   AttribHandler *head = m_attributes;
   while( head != 0 )
   {
      if ( head->attrib()->name() == attrib )
         return true;

      head =  head->next();
   }

   return false;
}


void CoreObject::reflectFrom( void *user_data )
{
   if ( m_cache == 0 )
      return;

   ObjectManager *mgr = m_generatedBy->getObjectManager();
   if( mgr != 0 && mgr->onObjectReflectFrom( this, user_data ) )
      return;

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


void CoreObject::reflectTo( void *user_data )
{
   if ( m_cache == 0 )
      return;

   ObjectManager *mgr = m_generatedBy->getObjectManager();
   if( mgr != 0 && mgr->onObjectReflectTo( this, user_data ) )
      return;

   const PropertyTable &pt = m_generatedBy->properties();

   for ( uint32 i = 0; i < pt.added(); i ++ )
   {
      const PropEntry &entry = pt.getEntry(i);
      if( entry.m_eReflectMode != e_reflectNone )
      {
         entry.reflectTo( this, user_data, m_cache[i] );
      }
   }
}


void CoreObject::gcMarkData( byte mark )
{
   if ( m_cache != 0 )
   {
      const PropertyTable &props = m_generatedBy->properties();
      for ( uint32 i = 0; i < props.added(); i ++ )
      {
         origin()->memPool()->markItemFast( m_cache[i] );
      }
   }

   if( m_user_data != 0 )
   {
      /*
         IF THIS ASSERT FAILS ---
         it means that the module where this object has been generated
         has not declared an object manager.

         [class]->getClassDef()->setObjectManager( [i.e. &Falcon::core_falcon_data_manager] );
      */
      fassert( m_generatedBy->getObjectManager() != 0 );
      m_generatedBy->getObjectManager()->onGarbageMark( origin(), m_user_data );
   }
}


bool CoreObject::isSequence() const
{
   if(  m_generatedBy->getObjectManager() != 0
        && m_generatedBy->getObjectManager()->isFalconData() )
      return static_cast< FalconData *>( m_user_data )->isSequence();
   return false;
}

void CoreObject::cacheStringProperty( const String& propName, const String &value )
{
   fassert( m_cache != 0 );
   uint32 pos;

   const PropertyTable &pt = m_generatedBy->properties();
   #if defined(NDEBUG)
      pt.findKey( propName, pos );
   #else
      bool found = pt.findKey( propName, pos );
      fassert( found );
   #endif

   Item &itm = m_cache[pos];
   if( itm.isString() )
      itm.asString()->bufferize( value );
   else
      itm = new GarbageString( origin(), value );
}



}

/* end of cobject.cpp */
