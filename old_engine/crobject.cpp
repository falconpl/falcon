/*
   FALCON - The Falcon Programming Language.
   FILE: crobject.cpp

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
#include <falcon/crobject.h>
#include <falcon/stream.h>

namespace Falcon {

CRObject::CRObject( const CoreClass *generator,  bool bDeserial ):
   CacheObject( generator, bDeserial )
{}

CRObject::CRObject( const CRObject &other ):
   CacheObject( other )
{}

bool CRObject::setProperty( const String &propName, const Item &value )
{
   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();
   if ( pt.findKey( propName, pos ) )
   {
      // to be optimized
      const PropertyTable &pt = m_generatedBy->properties();

      //Ok, we found the property, but what should we do with that?
      const PropEntry &entry = pt.getEntry( pos );
   
      // can we write it?
      if ( entry.m_bReadOnly ) {
         throw new AccessError( ErrorParam( e_prop_ro, __LINE__ ).extra( propName ) );
      }
   
      if ( entry.m_eReflectMode != e_reflectNone )
      {
         fassert( m_user_data != 0 || entry.m_eReflectMode == e_reflectSetGet );
         entry.reflectTo( this, m_user_data, value );
         // remember to cache the value.
      }
   
      if ( value.isReference() )
         m_cache[ pos ] = value;
      else
         *m_cache[ pos ].dereference() = value;
         
      return true;
   }

   return false;
}


bool CRObject::getProperty( const String &propName, Item &ret ) const
{
   fassert( m_generatedBy != 0 );

   register uint32 pos;
   const PropertyTable &pt = m_generatedBy->properties();

   if ( pt.findKey( propName, pos ) )
   {
      Item &cached = *m_cache[pos].dereference();
      const PropEntry &entry = pt.getEntry(pos);
      if ( entry.m_eReflectMode != e_reflectNone )
      {
         fassert( m_user_data != 0 || entry.m_eReflectMode == e_reflectSetGet );
         // this code allows to modify our cached value.
         entry.reflectFrom( const_cast<CRObject*>( this ), m_user_data, cached );
      }

      ret = cached;
      
      // already assigned, if possible
      return true;
   }

   return false;
}

CRObject *CRObject::clone() const
{
   return new CRObject( *this );
}

//============================================
CoreObject* CROpaqueFactory( const CoreClass *cls, void *user_data, bool bDeserial )
{
   CRObject* cro = new CRObject( cls, bDeserial );
   cro->setUserData( user_data );
   return cro;
}

CoreObject* CRFalconFactory( const CoreClass *cls, void *user_data, bool bDeserial )
{
   CRObject* cro = new CRObject( cls, bDeserial );
   if( user_data != 0 )
      cro->setUserData( static_cast<FalconData*>(user_data) );
   return cro;
}

CoreObject* CRSequenceFactory( const CoreClass *cls, void *user_data, bool bDeserial )
{
   CRObject* cro = new CRObject( cls, bDeserial );
   if( user_data != 0 )
      cro->setUserData( static_cast<Sequence*>(user_data) );
   return cro;
}

} // namespace Falcon

/* end of crobject.cpp */
