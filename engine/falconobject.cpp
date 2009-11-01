/*
   FALCON - The Falcon Programming Language.
   FILE: falconobject.cpp

   Falcon Object - Standard instance of classes in script
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom dic 5 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon Object - Standard instance of classes in script
*/

#include <falcon/falconobject.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/falcondata.h>

namespace Falcon {

FalconObject::FalconObject( const CoreClass *generator, bool bSeralizing ):
   CacheObject( generator, bSeralizing )
{}

FalconObject::FalconObject( const FalconObject &other ):
   CacheObject( other )
{}

FalconObject::~FalconObject()
{}

FalconObject *FalconObject::clone() const
{
   // create and ask not to create the base copy of the item table.
   FalconObject* fo = new FalconObject( *this );

   // was the inner data uncloneable? -- this is a rare case.
   if( m_user_data != 0 && fo->m_user_data == 0 )
      return 0; // and let the gc take care of the Falcon Object

   return fo;
}

CoreObject* OpaqueObjectFactory( const CoreClass *cls, void *data, bool bDeserializing )
{
   CoreObject* co =  new FalconObject( cls, bDeserializing );
   co->setUserData( data );
   return co;
}

CoreObject* FalconObjectFactory( const CoreClass *cls, void *data, bool bDeserializing )
{
   CoreObject* co =  new FalconObject( cls, bDeserializing );
   if ( data != 0 )
      co->setUserData( static_cast<FalconData* >(data) );
   return co;
}

CoreObject* FalconSequenceFactory( const CoreClass *cls, void *data, bool bDeserializing )
{
   CoreObject* co =  new FalconObject( cls, bDeserializing );
   if ( data != 0 )
      co->setUserData( static_cast<Sequence* >(data) );
   return co;
}

} // namespace Falcon

/* end of falconobject.cpp */
