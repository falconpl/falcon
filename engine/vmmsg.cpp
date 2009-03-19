/*
   FALCON - The Falcon Programming Language.
   FILE: vmmsg.cpp

   Asynchronous message for the Virtual Machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Feb 2009 16:08:50 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Asynchronous message for the Virtual Machine - Implementation.
*/

#include <falcon/vmmsg.h>
#include <falcon/globals.h>
#include <falcon/vm.h>
#include <falcon/garbagelock.h>

#define PARAMS_GROWTH   8

namespace Falcon {


VMMessage::VMMessage( VMachine* target, const String &msgName ):
   m_msg( msgName ),
   m_params(0),
   m_allocated(0),
   m_pcount(0),
   m_next(0),
   m_target( target )
{
}


VMMessage::~VMMessage()
{
   if( m_params != 0 )
   {
      for(uint32 i = 0; i < m_pcount; ++i )
      {
         m_target->unlock( m_params[i] );
      }
      memFree( m_params );
   }
}


void VMMessage::addParam( const Item &itm )
{
   if ( m_params == 0 )
   {
      m_params = (GarbageLock **) memAlloc( sizeof( GarbageLock* ) * PARAMS_GROWTH );
      m_allocated = PARAMS_GROWTH;
   }
   else if( m_pcount == m_allocated ) {
      m_allocated += PARAMS_GROWTH;
      m_params = (GarbageLock **) memRealloc( m_params, sizeof( GarbageLock* ) * m_allocated );
   }
   m_params[ m_pcount++ ] = m_target->lock( itm );
}


Item *VMMessage::param( uint32 p ) const
{
   return p < m_pcount ? &m_params[p]->item() : 0; 
}


void VMMessage::onMsgComplete( bool )
{
}

}

/* end of vmmsg.cpp */
