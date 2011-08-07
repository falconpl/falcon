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

namespace Falcon {

UserCarrier::UserCarrier( uint32  itemcount ):
   m_data( new Item[ itemcount ] ),
   m_itemCount( itemcount ),
   m_gcMark(0)
{
   for ( size_t i = 0; i < itemcount; ++i )
   {
      m_data[i].setNil();
   }  
}


UserCarrier::UserCarrier( const UserCarrier& other ):
   m_data( new Item[ other.m_itemCount ] ),
   m_itemCount( other.m_itemCount ),
   m_gcMark(0)
{
   for ( size_t i = 0; i < m_itemCount; ++i )
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
   for ( size_t i = 0; i < m_itemCount; ++i )
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


UserCarrier* UserCarrier::clone()
{
   return new UserCarrier( *this );
}
   
}

/* end of usercarrier.cpp */
