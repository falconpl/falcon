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

CoreObject::CoreObject( VMachine *vm, const PropertyTable &original, Symbol *inst ):
   Garbageable( vm, sizeof( this ) + sizeof( void * ) * 2 * original.size() ),
   m_properties( original ),
   m_instanceOf( inst ),
   m_attributes( 0 ),
   m_user_data( 0 ),
   m_user_data_shared( true )  // not true, but avoids a check in destructor
{
   // duplicate the strings in the property list
   for ( uint32 i = 0 ; i < m_properties.size(); i ++ )
   {
      Item *itm = m_properties.getValue( i );
      if( itm->isString() )
      {
         String *s = new GarbageString( vm, *itm->asString() );
         itm->setString( s );
      }
   }
}


CoreObject::CoreObject( VMachine *vm, UserData *ud ):
   Garbageable( vm, sizeof( this ) ),
   m_instanceOf( 0 ),
   m_properties( 0 ),
   m_attributes( 0 ),
   m_user_data( ud ),
   m_user_data_shared( ud->shared() )
{
}


CoreObject::~CoreObject()
{
   if ( ! m_user_data_shared )
      delete m_user_data;

   while( m_attributes != 0 )
   {
      // removing the attribute from this will also cause m_attributes to move forward
      m_attributes->attrib()->removeFrom( this );
   }
}


bool CoreObject::derivedFrom( const String &className ) const
{
   if ( m_instanceOf == 0 )
      return false;

   if ( m_instanceOf->name() == className )
   {
      return true;
   }

   // else search the base class name in the inheritance properties.
   Item temp;
   if ( getProperty( className, temp ) )
   {
      Item *itm = temp.dereference();

      if ( itm->isClass() )
         return true;
   }

   return false;
}


bool CoreObject::setPropertyRef( const String &propName, Item &value )
{
   register uint32 pos;

   if ( m_properties.findKey( &propName, pos ) )
   {
      Item *prop = m_properties.getValue( pos );
      origin()->referenceItem( *prop, value );
      if( m_user_data != 0 && m_user_data->isReflective() )
      {
         m_user_data->setProperty( origin(), propName, *prop );
      }

      return true;
   }
   // no reflection for property by reference.

   return false;
}


bool CoreObject::setProperty( const String &propName, const Item &value )
{
   register uint32 pos;

   if ( m_properties.findKey( &propName, pos ) )
   {
      Item *prop = m_properties.getValue( pos );

      *prop->dereference() = value;

      if( m_user_data != 0 && m_user_data->isReflective() )
      {
         m_user_data->setProperty( origin(), propName, *prop );
      }

      return true;
   }

   return false;
}


bool CoreObject::setProperty( const String &propName, const String &value )
{
   register uint32 pos;

   if ( m_properties.findKey( &propName, pos ) )
   {
      Item *prop = m_properties.getValue( pos );

      String *str = new GarbageString( origin(), value );
      prop->dereference()->setString( str );

      if( m_user_data != 0 && m_user_data->isReflective() )
      {
         m_user_data->setProperty( origin(), propName, *prop );
      }

      return true;
   }

   return false;
}


bool CoreObject::setProperty( const String &propName, int64 value )
{
   register uint32 pos;

   if ( m_properties.findKey( &propName, pos ) )
   {
      Item *prop = m_properties.getValue( pos );
      prop->dereference()->setInteger( value );

      if( m_user_data != 0 && m_user_data->isReflective() )
      {
         m_user_data->setProperty( origin(), propName, *prop );
      }

      return true;
   }

   return false;
}


bool CoreObject::getProperty( const String &propName, Item &ret ) const
{
   register uint32 pos;

   if( m_properties.findKey( &propName, pos ) )
   {
      Item *prop = m_properties.getValue( pos );
      if( m_user_data != 0 && m_user_data->isReflective() )
      {
         m_user_data->getProperty( origin(), propName, *prop );
      }

      ret = *prop;
      return true;
   }


   return false;
}


Item *CoreObject::getProperty( const String &propName )
{
   register uint32 pos;
   if( ! m_properties.findKey( &propName, pos ) )
      return 0;

   Item *prop = m_properties.getValue( pos );

   if( m_user_data != 0 && m_user_data->isReflective() )
   {
      m_user_data->getProperty( origin(), propName, *prop );
   }

   return prop;
}


bool CoreObject::getMethod( const String &propName, Item &method ) const
{
   register uint32 pos;
   if( ! m_properties.findKey( &propName, pos ) )
      return false;

   Item *mth = m_properties.getValue( pos );

   if( m_user_data != 0 && m_user_data->isReflective() )
   {
      m_user_data->getProperty( origin(), propName, *mth );
   }
   method = *mth;

   return method.methodize( this );
}


Item &CoreObject::getPropertyAt( uint32 pos ) const
{
   register Item *prop = m_properties.getValue( pos );
   if( m_user_data && m_user_data->isReflective() )
   {
      m_user_data->getProperty( origin(), *m_properties.getKey( pos ), *prop );
   }
   return *prop;
}


CoreObject *CoreObject::clone() const
{
   UserData *ud = 0;

   // if we can't clone the user data, we should just forbid cloning this object.
   if ( m_user_data != 0 )
   {
      ud = m_user_data->clone();
      if( ud == 0 )
         return 0;
   }

   CoreObject *other = new CoreObject( origin(), this->m_properties, m_instanceOf );

   // copy attribute list
   AttribHandler *head = m_attributes;
   while( head != 0 )
   {
      head->attrib()->giveTo( other );
      head = head->next();
   }

   other->setUserData( ud );

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


bool CoreObject::configureTo( void *data )
{
   if ( ! m_properties.hasConfig() )
      return false;

   byte *pData = (byte *) data;
   for ( uint32 i = 0; i < m_properties.added(); i ++ )
   {
      const PropertyTable::config &cfg = m_properties.getConfig( i );
      int64 value = m_properties.getValue( i )->forceInteger();
      switch( cfg.m_size )
      {
         case 1:
            pData[ cfg.m_offset ] = (byte) value;
            break;

         case 2:
            if ( cfg.m_isSigned )
               *((int16 *) (pData + cfg.m_offset) ) = (int16) value;
            else
               *((uint16 *) (pData + cfg.m_offset) ) = (uint16) value;
            break;

         case 4:
            if ( cfg.m_isSigned )
               *((int32 *) (pData + cfg.m_offset) ) = (int32) value;
            else
               *((uint32 *) (pData + cfg.m_offset) ) = (uint32) value;
           break;

         case 8:
            if ( cfg.m_isSigned )
               *((int64 *) (pData + cfg.m_offset) ) = value;
            else
               *((uint64 *) (pData + cfg.m_offset) ) = m_properties.getValue( i )->forceInteger();
            break;
      }
   }

   return true;
}


bool CoreObject::configureFrom( void *data )
{
   if ( ! m_properties.hasConfig() )
      return false;

   byte *pData = (byte *) data;
   for ( uint32 i = 0; i < m_properties.added(); i ++ )
   {
      const PropertyTable::config &cfg = m_properties.getConfig( i );
      switch( cfg.m_size )
      {
         case 1:
            m_properties.getValue( i )->setInteger( pData[ cfg.m_offset ] );
            break;

         case 2:
            if ( cfg.m_isSigned )
               m_properties.getValue( i )->setInteger( *((int16 *) (pData + cfg.m_offset) ) );
            else
               m_properties.getValue( i )->setInteger( *((uint16 *) (pData + cfg.m_offset) ) );
            break;

         case 4:
            if ( cfg.m_isSigned )
               m_properties.getValue( i )->setInteger( *((int32 *) (pData + cfg.m_offset) ) );
            else
               m_properties.getValue( i )->setInteger( *((uint32 *) (pData + cfg.m_offset) ) );
            break;

         case 8:
            if ( cfg.m_isSigned )
               m_properties.getValue( i )->setInteger( *((int64 *) (pData + cfg.m_offset) ) );
            else
               m_properties.getValue( i )->setInteger( *((uint64 *) (pData + cfg.m_offset) ) );
            break;
      }
   }

   return true;
}

}

/* end of cobject.cpp */
