/*
   The Falcon Programming Language
   FILE: dbus_ext.cpp

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
   Interface extension functions
*/

#include <falcon/engine.h>
#include "dbus_mod.h"
#include "dbus_ext.h"
#include "dbus_st.h"

namespace Falcon {
namespace Ext {

class VarParsStruct
{
   byte *m_pData;
   AutoCString **m_vcs;

   int32 m_strings;
   int32 m_strAlloc;
   int32 m_current;
   int32 m_alloc;
public:
   
   VarParsStruct():
      m_pData( 0 ),
      m_vcs ( 0 ),
      m_strings( 0 ),
      m_strAlloc( 0 ),
      m_current( 0 ),
      m_alloc( 0 )
   {}
   
   ~VarParsStruct()
   {
      if ( m_pData != 0 )
         memFree( m_pData );
      
      if ( m_strings > 0 )
      {
         for ( int32 i = 0; i < m_strings; ++i )
            delete m_vcs[i];
         memFree( m_vcs );
      }
   }
   
   void* addInt32( int32 data )
   {
      if ( m_current + sizeof( int32 ) > m_alloc )
      {
         m_alloc += 16 * sizeof( int32 );
         m_pData = (byte*) memRealloc( m_pData, m_alloc );
      }
      
      int32 *p = ((int32*)(m_pData+m_current ) );
      m_current += sizeof( int32 );
      *p = data;
      return p;
   }
   
   
   void* addInt64( int64 data )
   {
      if ( m_current + sizeof( int64 ) > m_alloc )
      {
         m_alloc += 16 * sizeof( int32 );
         m_pData = (byte*) memRealloc( m_pData, m_alloc );
      }
      
      int64 *p = ((int64*)(m_pData+m_current) );
      m_current += sizeof( int64 );
      *p = data;
      return p;
   }
   
   void* addNumeric( numeric data )
   {
      if ( m_current + sizeof( numeric ) > m_alloc )
      {
         m_alloc += 16 * sizeof( int32 );
         m_pData = (byte*) memRealloc( m_pData, m_alloc );
      }
      
      numeric *p = ((numeric*)(m_pData+m_current) );
      m_current += sizeof( numeric );
      *p = data;
      return p;
   }
   
   void* addString( const String &data )
   {
      if ( m_current + sizeof( char* ) > m_alloc )
      {
         m_alloc += 16 * sizeof( int32 );
         m_pData = (byte*) memRealloc( m_pData, m_alloc );
      }
      
      const char **p = ((const char**)(m_pData+m_current) );
      m_current += sizeof( char* );
      
      // now convert it to a string.
      if( m_strings >= m_strAlloc )
      {
         m_strAlloc += 8;
         m_vcs = (AutoCString**) memRealloc( m_vcs, m_strAlloc * sizeof( AutoCString* ) ); 
      }
      m_vcs[ m_strings ] = new AutoCString( data );
      *p = m_vcs[ m_strings ]->c_str();
      m_strings++;
      return p;
   }

};

// The following is a faldoc block for the function
/*#
   @class DBus
   @brief Generic interface to the DBUS system.

   TODO
*/

/*#
   @init DBus
   @brief Opens the connection to the DBUS system.
   @raise DBusError in case of failure.
*/

FALCON_FUNC  DBus_init( VMachine *vm )
{
   // create the new instance
   Mod::DBusWrapper* wrapper = new Mod::DBusWrapper;
   if ( ! wrapper->connect() )
   {
      vm->raiseModError( new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE, __LINE__ )
         .desc( wrapper->error()->name )
         .extra( wrapper->error()->message ) ) );
      
      delete wrapper;
      return;
   }
   
   vm->self().asObject()->setUserData( wrapper );
}


//====================
// Utility function
static bool s_append_param( VMachine *vm, const Item &src, DBusMessageIter &args, VarParsStruct &vps )
{
   bool bSuccess;
   
   switch( src.type() )
   {
      case FLC_ITEM_INT:
         if ( src.asInteger() < 0x7FFFFFFF && src.asInteger() > -0x7FFFFFFF )
            bSuccess = dbus_message_iter_append_basic( &args, DBUS_TYPE_INT32, vps.addInt32(src.asInteger()) );
         else
            bSuccess = dbus_message_iter_append_basic( &args, DBUS_TYPE_INT64, vps.addInt64(src.asInteger()) );
         break;
         
      case FLC_ITEM_NUM:
         bSuccess = dbus_message_iter_append_basic( &args, DBUS_TYPE_DOUBLE, vps.addNumeric( src.asNumeric() ) );
         break;
      
      case FLC_ITEM_STRING:
         bSuccess = dbus_message_iter_append_basic( &args, DBUS_TYPE_STRING, vps.addString( *src.asString() ) );
         break;
      
      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
            extra( "S,S,S,[...]" ) ) );
         return false;
   }
   
   if (! bSuccess ) 
   { 
      vm->raiseModError( new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) ) );
      return false;
   }
   
   return true;
}


/*#
   @method signal DBus
   @brief Broadcast a message directed to all the potential DBUS listeners.
   @param path the path to the object emitting the signal 
   @param interface the interface the signal is emitted from
   @param name name of the signal 
   @param ... Parameters for the signal.
   @raise DBusError in case of failure.
   
*/
FALCON_FUNC  DBus_signal( VMachine *vm )
{
   Item *i_path = vm->param(0);
   Item *i_interface = vm->param(1);
   Item *i_name = vm->param(2);
   
   if( i_path == 0 || ! i_path->isString() ||
       i_interface == 0  || ! i_interface->isString() ||
       i_name == 0 || ! i_interface->isString() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
         extra( "S,S,S,[...]" ) ) );
      return;
   }
   
   // get the connection
   Mod::DBusWrapper* wp = static_cast<Mod::DBusWrapper*>( vm->self().asObject()->getUserData() );
   
   // get the params in string format
   AutoCString cpath( *i_path->asString() );
   AutoCString ciface( *i_interface->asString() );
   AutoCString cname( *i_name->asString() );
   
   // create a signal and check for errors 
   DBusMessage* msg = dbus_message_new_signal( cpath.c_str(), // object name of the signal
         ciface.c_str(), // interface name of the signal
         cname.c_str() ); // name of the signal
         
   if ( msg == 0 )
   { 
      vm->raiseModError( new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) ) );
      return;
   }
   
   dbus_uint32_t serial = 0; // unique number to associate replies with requests
   
   // this structure must exist until we trash the message out of the interface.
   VarParsStruct vps;
   
   if( vm->paramCount() > 3)
   {
      
      DBusMessageIter args;
      // append arguments onto signal
      dbus_message_iter_init_append(msg, &args);
   
      for( int pid = 3; pid < vm->paramCount(); ++pid )
      {
         Item *i_param = vm->param(  pid );
         // in case of failure, we can just return
         if ( ! s_append_param( vm, *i_param, args, vps ) )
         {
            // free the message 
            dbus_message_unref(msg);
            return;
         }
      }
   }

   // send the message and flush the connection
   if (!dbus_connection_send( wp->conn(), msg, &serial)) 
   {
      vm->raiseModError( new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) ) );
      
      // free the message 
      dbus_message_unref(msg);
      return;
   }
   
   dbus_connection_flush(wp->conn()); 
   
   // free the message 
   dbus_message_unref(msg);
}


//======================================================
// DynLib error
//======================================================

/*#
   @class DBusError
   @optparam code Error code
   @optparam desc Error description
   @optparam extra Extra description of specific error condition.

   @from Error( code, desc, extra )
   @brief DBus specific error.

   Inherited class from Error to distinguish from a standard Falcon error.
*/

/*#
   @init DBusError
   See Core Error class description.
*/
FALCON_FUNC DBusError_init( VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Mod::f_DBusError );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of dbus_mod.cpp */
