/*
   The Falcon Programming Language
   FILE: dbus_mod.cpp

   Falcon - DBUS official binding
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 17 Dec 2008 20:12:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: Giancarlo Niccolai

      Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

      http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

*/

/** \file
   Falcon - DBUS official binding
   Internal logic functions - implementation.
*/

#include "dbus_mod.h"

namespace Falcon {
namespace Mod {


DBusWrapper::DBusWrapper()
{
   m_content = new s_inner_data;
   m_content->m_refcount = 1;
   dbus_error_init( &m_content->m_err );
}

DBusWrapper::DBusWrapper( const DBusWrapper &other )
{
   other.m_content->m_refcount++;
   m_content = other.m_content;
}


DBusWrapper::~DBusWrapper()
{
   // Won't work great in MT, but we're working at that.
   if ( --m_content->m_refcount == 0 )
   {
      dbus_error_free( &m_content->m_err );
      delete m_content;
   }
}


bool DBusWrapper::connect()
{
   // connect to the bus
   m_content->m_conn = dbus_bus_get(DBUS_BUS_SESSION, &m_content->m_err );
   
   // If something went wrong, signal it
   if ( dbus_error_is_set(&m_content->m_err) || m_content->m_conn == 0 ) 
   { 
      return false;
   }
   
   return true;
}


FalconData* DBusWrapper::clone() const
{
   return new DBusWrapper( *this );
}

void DBusWrapper::gcMark( uint32 mk )
{
}

//=======================================================
//
//=======================================================

DBusPendingWrapper::DBusPendingWrapper( DBusConnection*c, DBusPendingCall* p ):
   m_conn( c ),
   m_pc( p )
{
   dbus_connection_ref( c );
   dbus_pending_call_ref( p );
}

DBusPendingWrapper::DBusPendingWrapper( const DBusPendingWrapper &other ):
   m_conn( other.m_conn ),
   m_pc( other.m_pc )
{
   dbus_connection_ref( m_conn );
   dbus_pending_call_ref( m_pc );
}

DBusPendingWrapper::~DBusPendingWrapper()
{
   dbus_connection_unref( m_conn );
   dbus_pending_call_unref( m_pc );
}

FalconData* DBusPendingWrapper::clone() const
{
   return new DBusPendingWrapper( *this );
}

void DBusPendingWrapper::gcMark( uint32 mk )
{
}

//=======================================================
//
//=======================================================

DBusMessageWrapper::DBusMessageWrapper( DBusMessage *msg ):
   m_msg( msg )
{
   dbus_message_ref( m_msg );
}

DBusMessageWrapper::DBusMessageWrapper( const DBusMessageWrapper &other ):
   m_msg ( other.m_msg )
{
   dbus_message_ref( m_msg );
}

DBusMessageWrapper::~DBusMessageWrapper( void )
{
   dbus_message_unref( m_msg );
}

FalconData *DBusMessageWrapper::clone( void ) const
{
   return new DBusMessageWrapper( *this );
}

void DBusMessageWrapper::gcMark( uint32 )
{
}

}
}


/* end of dbus_mod.cpp */
