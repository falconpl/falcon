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

#include <falcon/fassert.h>
#include <falcon/reflectobject.h>
#include <falcon/cclass.h>
#include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/falcondata.h>

namespace Falcon {

ReflectObject::ReflectObject( const CoreClass *generator,  void *user_data ):
   CoreObject( generator )
{
   if( user_data != 0 )
      setUserData( user_data );

   // In some controlled case, it may be the _init method to set the object.
}

ReflectObject::ReflectObject( const CoreClass* generator, FalconData* fdata ):
   CoreObject( generator )
{
   if( fdata != 0 )
      setUserData( fdata );
}

ReflectObject::ReflectObject( const CoreClass* generator, Sequence* seq ):
   CoreObject( generator )
{
   if( seq != 0 )
      setUserData( seq );
}

ReflectObject::ReflectObject( const ReflectObject& other ):
   CoreObject( other )
{}

ReflectObject::~ReflectObject()
{}


bool ReflectObject::setProperty( const String &propName, const Item &value )
{
   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   if ( pt.findKey( propName, pos ) )
   {
      const PropEntry &entry = pt.getEntry( pos );

      if ( entry.m_bReadOnly )
      {
         throw new AccessError( ErrorParam( e_prop_ro, __LINE__ ).extra( propName ) );
      }

      fassert( m_user_data != 0 );
      fassert( entry.m_eReflectMode != e_reflectNone );

      entry.reflectTo( this, m_user_data, value );
      return true;
   }

   return false;
}



bool ReflectObject::getProperty( const String &propName, Item &ret ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();

   if ( pt.findKey( propName, pos ) )
   {
      //Ok, we found the property, but what should we do with that?
      const PropEntry &entry = pt.getEntry( pos );
      fassert( entry.m_bReadOnly || entry.m_eReflectMode != e_reflectNone );

      if ( entry.m_bReadOnly && entry.m_eReflectMode == e_reflectNone )
      {
         ret = entry.m_value;
         return true;
      }

      fassert( m_user_data != 0 );
      entry.reflectFrom( const_cast<ReflectObject*>(this), m_user_data, ret );
      return true;
   }

   return false;
}

ReflectObject* ReflectObject::clone() const
{
   return new ReflectObject( *this );
}

//============================================
CoreObject* ReflectOpaqueFactory( const CoreClass *cls, void *user_data, bool )
{
  return new ReflectObject( cls, user_data );
}

CoreObject* ReflectFalconFactory( const CoreClass *cls, void *user_data, bool )
{
  return new ReflectObject( cls, static_cast<FalconData*>(user_data) );
}

CoreObject* ReflectSequenceFactory( const CoreClass *cls, void *user_data, bool )
{
  return new ReflectObject( cls, static_cast<Sequence*>(user_data) );
}

} // namespace Falcon

/* end of cobject.cpp */
