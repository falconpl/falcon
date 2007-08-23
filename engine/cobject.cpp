/*
   FALCON - The Falcon Programming Language.
   FILE: cobject.cpp
   $Id: cobject.cpp,v 1.11 2007/08/11 10:22:35 jonnymind Exp $

   Core object implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom dic 5 2004
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
   Core object implementation.
*/

#include <falcon/vm.h>
#include <falcon/item.h>
#include <falcon/cobject.h>
#include <falcon/symbol.h>

namespace Falcon {

CoreObject::CoreObject( VMachine *vm, const PropertyTable &original, uint64 attribs, Symbol *inst ):
   Garbageable( vm, sizeof( this ) + sizeof( void * ) * 2 * original.size() ),
   m_properties( original ),
   m_instanceOf( inst ),
   m_attributes( attribs ),
   m_user_data( 0 )
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
   m_user_data( ud )
{
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
         m_user_data->setProperty( propName, *prop );
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

      if ( value.isString() )
      {
         String *str = new GarbageString( origin(), *value.asString() );
         prop->dereference()->setString( str );
      }
      else
         *prop->dereference() = value;

      if( m_user_data != 0 && m_user_data->isReflective() )
      {
         m_user_data->setProperty( propName, *prop );
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
         m_user_data->setProperty( propName, *prop );
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
         m_user_data->setProperty( propName, *prop );
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
         m_user_data->getProperty( propName, *prop );
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
      m_user_data->getProperty( propName, *prop );
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
      m_user_data->getProperty( propName, *mth );
   }
   method = *mth;

   return method.methodize( this );
}

Item &CoreObject::getPropertyAt( uint32 pos ) const
{
   register Item *prop = m_properties.getValue( pos );
   if( m_user_data && m_user_data->isReflective() )
   {
      m_user_data->getProperty( *m_properties.getKey( pos ), *prop );
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

   CoreObject *other = new CoreObject( origin(), this->m_properties, m_attributes, m_instanceOf );
   other->m_user_data = ud;

   return other;
}

}

/* end of cobject.cpp */
