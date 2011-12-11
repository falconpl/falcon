/*
   FALCON - The Falcon Programming Language.
   FILE: usercarrier.cpp

   Base class for ClassUser based instance reflection.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 12:11:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/usercarrier.cpp"

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/item.h>
#include <falcon/usercarrier.h>
#include <falcon/errors/unsupportederror.h>

namespace Falcon {

UserCarrier::UserCarrier():
   m_data( 0 ),
   m_dataSize( 0 ),
   m_gcMark(0)
{}

UserCarrier::UserCarrier( uint32  itemcount ):
   m_data( itemcount == 0 ? 0 : new Item[ itemcount ] ),
   m_dataSize( itemcount ),
   m_gcMark(0)
{
   if( itemcount > 0 )
   {
      for ( size_t i = 0; i < itemcount; ++i )
      {
         m_data[i].setNil();
      }
   }
}


UserCarrier::UserCarrier( const UserCarrier& other ):
   m_data( new Item[ other.m_dataSize ] ),
   m_dataSize( other.m_dataSize ),
   m_gcMark(0)
{
   for ( size_t i = 0; i < m_dataSize; ++i )
   {
      m_data[i].setNil();
   }
}
   

UserCarrier::~UserCarrier()
{
   delete[] m_data;
}
   

void UserCarrier::gcMark( uint32 mark )   
{
   for ( size_t i = 0; i < m_dataSize; ++i )
   {
      Item& item = m_data[i];
      if( item.isGarbaged() )
      {
         Class* cls = 0;
         void* data = 0;
         item.asClassInst( cls, data );
         cls->gcMark( data, mark );
      }
   }
}
   
template<class _t>
UserCarrierT<_t>::UserCarrierT( const UserCarrierT<_t>& other ):
      UserCarrier( other )
{
   m_data = other.cloneData();
   if ( m_data == 0 )
   {
      throw new UnsupportedError( ErrorParam( e_uncloneable, __LINE__, SRC ).
         origin(ErrorParam::e_orig_runtime)
         );
   }
}

}

/* end of usercarrier.cpp */
