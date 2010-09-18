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

#include <stdio.h>

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
      delete wrapper;
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE, __LINE__ )
         .desc( wrapper->error()->name )
         .extra( wrapper->error()->message ) );
   }

   vm->self().asObject()->setUserData( wrapper );
}


//====================
// Utility functions
static Error* s_append_param( VMachine *vm, const Item &src, DBusMessageIter &args, VarParsStruct &vps )
{
   bool bSuccess;

   switch( src.type() )
   {
      case FLC_ITEM_BOOL:
         bSuccess = dbus_message_iter_append_basic( &args, DBUS_TYPE_BOOLEAN, vps.addInt32(src.asBoolean() ? 1 : 0 ) );
         break;

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
         return new ParamError( ErrorParam( e_inv_params ).
            extra( "S,S,S,[...]" ) );

   }

   if (! bSuccess )
   {
      return new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) );
   }

   return 0;
}


static bool s_extract_return( Item &target, DBusMessage *msg )
{
   DBusMessageIter args;

   if (!dbus_message_iter_init(msg, &args))
   {
      // no parameters
      return false;
   }
   
   CoreArray *arr = new CoreArray;
   Item item;

   // read the parameters
   do {
      
      switch( dbus_message_iter_get_arg_type(&args) )
      {
         case DBUS_TYPE_OBJECT_PATH:
         case DBUS_TYPE_STRING:
            {
            const char *v;
            dbus_message_iter_get_basic(&args, &v);
            item  = new CoreString;
            item.asString()->fromUTF8(v);
            }
            break;

         case DBUS_TYPE_BOOLEAN: {
            dbus_bool_t v;
            dbus_message_iter_get_basic(&args, &v);
            item.setBoolean( v );
            }
            break;

         case DBUS_TYPE_BYTE: {
            char v;
            dbus_message_iter_get_basic(&args, &v);
            item.setInteger( (byte) v );
            }
            break;

         case DBUS_TYPE_INT16: {
            dbus_int16_t v;
            dbus_message_iter_get_basic(&args, &v);
            item.setInteger( v );
            }
            break;

         case DBUS_TYPE_UINT16: {
            dbus_uint16_t v;
            dbus_message_iter_get_basic(&args, &v);
            item.setInteger( v );
            }
            break;

         case DBUS_TYPE_INT32: {
            dbus_int32_t v;
            dbus_message_iter_get_basic(&args, &v);
            item.setInteger( v );
            }
            break;

         case DBUS_TYPE_UINT32: {
            dbus_uint32_t v;
            dbus_message_iter_get_basic(&args, &v);
            item.setInteger( v );
            }
            break;

         case DBUS_TYPE_INT64: {
            dbus_int64_t v;
            dbus_message_iter_get_basic(&args, &v);
            item.setInteger( v );
            }
            break;

         case DBUS_TYPE_UINT64: {
            dbus_uint64_t v;
            dbus_message_iter_get_basic(&args, &v);
            item.setInteger( v );
            }
            break;

         case DBUS_TYPE_DOUBLE: {
            double v;
            dbus_message_iter_get_basic(&args, &v);
            item.setNumeric( v );
            }
            break;

         //case DBUS_TYPE_ARRAY:
         default:
            return false;
      }
      arr->append( item );

   } while( dbus_message_iter_next( &args ) );
   
   target = arr;
   return true;
}


/*#
   @method signal DBus
   @brief Broadcast a message directed to all the potential DBUS listeners.
   @param path the path from the object emitting the signal
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
      throw new ParamError( ErrorParam( e_inv_params ).
         extra( "S,S,S,[...]" ) );
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
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) );
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
         Error* error;
         if ( (error = s_append_param( vm, *i_param, args, vps )) != 0 )
         {
            // free the message
            dbus_message_unref(msg);
            throw error;
         }
      }
   }

   // send the message and flush the connection
   if (!dbus_connection_send( wp->conn(), msg, &serial))
   {
      dbus_message_unref(msg);
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) );
   }

   dbus_connection_flush(wp->conn());

   // free the message
   dbus_message_unref(msg);
}

/*#
   @method invoke DBus
   @brief Invoke a remote DBUS method.
   @param destination Well known name of the service provider where the method is searched.
   @param path Path to the object in the service provider.
   @param interface Interface in which the method is searched.
   @param name Method name.
   @param ... Parameters for the method invocation.
   @return an instance of @a DBusPendingCall class.
   @raise DBusError in case of failure.

   Use the returned instance to wait for a reply.
*/
FALCON_FUNC  DBus_invoke( VMachine *vm )
{
   Item *i_target = vm->param(0);
   Item *i_path = vm->param(1);
   Item *i_interface = vm->param(2);
   Item *i_name = vm->param(3);

   if( i_target == 0 || ! i_target->isString() ||
       i_path == 0 || ! i_path->isString() ||
       i_interface == 0  || ! i_interface->isString() ||
       i_name == 0 || ! i_interface->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
         extra( "S,S,S,S,[...]" ) );
   }

   // get the connection
   Mod::DBusWrapper* wp = static_cast<Mod::DBusWrapper*>( vm->self().asObject()->getUserData() );

   // get the params in string format
   AutoCString ctarget( *i_target->asString() );
   AutoCString cpath( *i_path->asString() );
   AutoCString ciface( *i_interface->asString() );
   AutoCString cname( *i_name->asString() );

   DBusMessage* msg = dbus_message_new_method_call(
         ctarget.c_str(), // target for the method call
         cpath.c_str(), // object to call on
         ciface.c_str(), // interface to call on
         cname.c_str()); // method name

   if ( msg == 0 )
   {
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) );
   }

   // append arguments
   // this structure must exist until we trash the message out of the interface.
   VarParsStruct vps;

   if( vm->paramCount() > 4)
   {

      DBusMessageIter args;
      // append arguments onto signal
      dbus_message_iter_init_append(msg, &args);

      for( int pid = 4; pid < vm->paramCount(); ++pid )
      {
         Item *i_param = vm->param(  pid );
         // in case of failure, we can just return
         Error* error;
         if ( (error = s_append_param( vm, *i_param, args, vps )) != 0 )
         {
            // free the message
            dbus_message_unref(msg);
            throw error;
         }
      }
   }

   DBusPendingCall* pending;

   // send message and get a handle for a reply
   if (!dbus_connection_send_with_reply ( wp->conn(), msg, &pending, -1) || pending == 0 ) {
                                                           // -1 is default timeout
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+1, __LINE__ )
         .desc( FAL_STR( dbus_out_of_mem ) ) );
      dbus_message_unref(msg);
   }

   // send the data
   dbus_connection_flush(wp->conn());

   // free message
   dbus_message_unref(msg);

   // return the pending connection
   Item* i_cls = vm->findWKI( "%DBusPendingCall" );
   fassert( i_cls != 0 && i_cls->isClass() );
   CoreClass* cls = i_cls->asClass();
   CoreObject* obj = cls->createInstance();
   obj->setUserData( new Mod::DBusPendingWrapper( wp->conn(), pending ) );
   vm->retval( obj );
}


/*#
   @method dispatch DBus
   @brief Perform a dispatch loop on message queues.
   @optparam timeout An optional timeout to be idle for messages to be sent or receive.
   @raise DBusError in case of failure.

   Set @b timeout to zero (or empty) to just dispatch ready messages, or to -1 to wait forever.
   Otherwise, waits for seconds and fractions.
*/
FALCON_FUNC  DBus_dispatch( VMachine *vm )
{
   Item *i_timeout = vm->param(0);

   if( i_timeout != 0 && ! i_timeout->isOrdinal() )
   {
      new ParamError( ErrorParam( e_inv_params ).
         extra( "[N]" ) );
   }
   
   int to = ( i_timeout ) ? ( i_timeout->forceNumeric() * 1000.0 ) : 0;

   // get the connection
   Mod::DBusWrapper* wp = static_cast<Mod::DBusWrapper*>( vm->self().asObject()->getUserData() );
   dbus_connection_read_write_dispatch( wp->conn(), to );
}


/*#
   @method popMessage DBus
*/
FALCON_FUNC  DBus_popMessage( VMachine *vm )
{
   Mod::DBusWrapper *wp = static_cast< Mod::DBusWrapper* >( vm->self().asObject()->getUserData() );
   DBusMessage *msg;
   
   msg = dbus_connection_pop_message( wp->conn() );
   
   if ( msg == 0 )
   {
      vm->retnil();
   }
   else
   {
      Item* i_cls = vm->findWKI( "%DBusMessage" );
      fassert( i_cls != 0 && i_cls->isClass() );
      CoreClass* cls = i_cls->asClass();
      CoreObject* obj = cls->createInstance();
      obj->setUserData( new Mod::DBusMessageWrapper( msg ) );
      vm->retval( obj );
   }
}
   
   

/*#
   @method addMatch DBus
   @brief Adds an active filter for incoming signals.
   @optparam rule The filter rule in DBUS rule specification format.
   @raise DBusError in case of failure.

   Set @b rule to nil or leave empty to perform a "full filter" request.

   See the dbus_bus_add_match() description in the official DBUS low
   level documentation.
*/
FALCON_FUNC  DBus_addMatch( VMachine *vm )
{
   Item *i_rule = vm->param(0);

   if( i_rule != 0 && ! ( i_rule->isString() || i_rule->isNil() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
         extra( "[S]" ) );
   }

   Mod::DBusWrapper* wp = static_cast<Mod::DBusWrapper*>( vm->self().asObject()->getUserData() );

   if( i_rule != 0 && ! i_rule->isNil() )
   {
      AutoCString cs( *i_rule->asString() );

      dbus_bus_add_match( wp->conn(),
         cs.c_str(),
         wp->error());
   }
   else {
      dbus_bus_add_match( wp->conn(),
         0,
         wp->error());
   }

   dbus_connection_flush( wp->conn() );
   if ( dbus_error_is_set( wp->error() ) )
   {
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE, __LINE__ )
         .desc( wp->error()->name )
         .extra( wp->error()->message ) );
   }
}


/*#
   @method removeMatch DBus
   @brief Removes an active filter for incoming signals.
   @optparam rule The filter rule in DBUS rule specification format.
   @raise DBusError in case of failure.

   Set @b rule to nil or leave empty to remove a previous "full filter" request.

   See the dbus_bus_remove_match() description in the official DBUS low
   level documentation.
*/
FALCON_FUNC  DBus_removeMatch( VMachine *vm )
{
   Item *i_rule = vm->param(0);

   if( i_rule != 0 && ! ( i_rule->isString() || i_rule->isNil() ) )
   {
      new ParamError( ErrorParam( e_inv_params ).
         extra( "[S]" ) );
   }

   Mod::DBusWrapper* wp = static_cast<Mod::DBusWrapper*>( vm->self().asObject()->getUserData() );

   if( i_rule != 0 && ! i_rule->isNil() )
   {
      AutoCString cs( *i_rule->asString() );

      dbus_bus_remove_match( wp->conn(),
         cs.c_str(),
         wp->error());
   }
   else {
      dbus_bus_remove_match( wp->conn(),
         0,
         wp->error());
   }

   dbus_connection_flush( wp->conn() );
   if ( dbus_error_is_set( wp->error() ) )
   {
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE, __LINE__ )
         .desc( wp->error()->name )
         .extra( wp->error()->message ) );
   }

}

FALCON_FUNC  DBus_requestName( VMachine *vm )
{
   Item *i_name  = vm->param( 0 );
   Item *i_flags = vm->param( 1 );
   
   if ( i_name  == 0 || ! i_name->isString() ||
        i_flags == 0 || ! i_flags->isInteger() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "[S,N]" ) );
   }
 
   Mod::DBusWrapper *wp = static_cast< Mod::DBusWrapper* >( vm->self().asObject()->getUserData() );
   AutoCString name( i_name->asString() );
   int flags = i_flags->asInteger();
   
   int res = dbus_bus_request_name( wp->conn(), name.c_str(), flags, wp->error() );
   
   if ( dbus_error_is_set( wp->error() ) )
   {
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE, __LINE__ )
         .desc( wp->error()->name )
         .extra( wp->error()->message ) );
   }
   
   vm->retval( res );
}

static DBusHandlerResult s_dbusHandler( DBusConnection *conn, DBusMessage *msg, void *userData )
{
   Mod::DBusHandlerData *data = static_cast< Mod::DBusHandlerData* >( userData );
   AutoCString name( data->name );
   AutoCString interface( data->interface );
   
   if (( data->isSignal && dbus_message_is_signal( msg, interface.c_str(), name.c_str() ) ) ||
         dbus_message_is_method_call( msg, interface.c_str(), name.c_str() ) )
   {
      Item params;
      s_extract_return( params, msg );
      data->vm->pushParameter( params );
      data->vm->callItem( data->handler, 1 );
      return DBUS_HANDLER_RESULT_HANDLED;
   }
   
    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}
   

FALCON_FUNC  DBus_addFilter( VMachine *vm )
{
   Item *i_interface = vm->param( 0 );
   Item *i_name      = vm->param( 1 );
   Item *i_handler   = vm->param( 2 );
   Item *i_isSignal  = vm->param( 3 );
   
   if ( i_interface == 0 || ! i_interface->isString() ||
        i_name == 0      || ! i_name->isString() ||
        i_handler == 0    || ! i_handler->isFunction() ||
        i_isSignal != 0  && ! i_isSignal->isBoolean() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "[S,S,C,[B]]" )
      );
   }
   
   Mod::DBusWrapper *wp = static_cast< Mod::DBusWrapper* >( vm->self().asObject()->getUserData() );
   Mod::DBusHandlerData *data = static_cast< Mod::DBusHandlerData* >(memAlloc( sizeof(Mod::DBusHandlerData ) ));
   
   data->vm = vm;
   data->interface = i_interface->asString();
   data->name = i_name->asString();
   data->handler = i_handler->asFunction();
   data->isSignal = (i_isSignal) ? i_isSignal->isTrue() : true;
   
   dbus_connection_add_filter( wp->conn(), s_dbusHandler, (void *)data, memFree );
}


//============================================================
// Pending call
//

/*#
   @class DBusPendingCall
   @brief Handle for currently open method calls.

   This class is returned by @a DBus.invoke and cannot be directly instantiated.
*/

/*#
   @method wait DBusPendingCall
   @brief Wait for a pending call to complete and returns the remote method result.
   @return An item or an array of items returned by the remote method.
   @raise DBusError if the method call couldn't be performed, of if the remote
   side returned an error.

   This method is blocking (and currently not respecting VM interruption protocol).
*/
FALCON_FUNC  DBusPendingCall_wait( VMachine *vm )
{
   Mod::DBusPendingWrapper* wrapper = static_cast<Mod::DBusPendingWrapper*>( vm->self().asObject()->getUserData() );
   DBusPendingCall* pending = wrapper->pending();

   // block until we receive a reply
   dbus_pending_call_block(pending);

   // get the reply message
   DBusMessage* msg = dbus_pending_call_steal_reply(pending);
   if ( msg == 0 )
   {
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+2, __LINE__ )
         .desc( FAL_STR( dbus_null_reply ) ) );
      dbus_message_unref(msg);
      return;
   }

   // did the method call was errorful?
   if( dbus_message_get_type( msg ) == DBUS_MESSAGE_TYPE_ERROR )
   {
      String resDesc = dbus_message_get_error_name( msg );
      resDesc;

      Item temp;
      if ( s_extract_return( temp, msg ) && temp.isString() )
      {
         resDesc += ":";
         resDesc += *temp.asString();
      }

      dbus_message_unref(msg);
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+4, __LINE__ )
         .desc( FAL_STR( dbus_method_call ) )
         .extra( resDesc ) );
   }

   // free the pending message handle
   //dbus_pending_call_unref(pending);
   vm->regA().setNil();
   bool res = s_extract_return( vm->regA(), msg );

   // free reply and close connection
   dbus_message_unref(msg);

   if( ! res )
   {
      throw new Mod::f_DBusError( ErrorParam( FALCON_ERROR_DBUS_BASE+3, __LINE__ )
               .desc( FAL_STR( dbus_unknown_type ) ) );
   }
}


/*#
   @method completed DBusPendingCall
   @brief Checks if a pending call has completed.
   @optparam dispatch set to true to force dispatching of messages (and state refresh).
   @return True if the pending call can be waited on without blocking.

   This method can be used to poll periodically to see if an answer has come in the
   meanwhile.

   If the @b dispatch parameter is not specified, or if it's false, the network is not
   read again for new incoming data on the DBUS connection. This means a @a DBus.dispatch
   method or other DBUS operations must be performed elsewhere for this pending call to be
   updated and eventually completed. For example:

   @code
      while pending.completed()
         ...
         sleep(...)
         conn.dispatch()
      end
   @endcode


   If the parameter is set to true a single dispatch loop
   is performed too. Usually, it takes at least 2 dispatch loops to receive a complete answer.

   @code
      while pending.completed( true )
         ...
         sleep(...)
         // no need for conn.dispatch() to be called
      end
   @endcode

*/
FALCON_FUNC  DBusPendingCall_completed( VMachine *vm )
{
   Item *i_dispatch =  vm->param(0);

   Mod::DBusPendingWrapper* wrapper = static_cast<Mod::DBusPendingWrapper*>( vm->self().asObject()->getUserData() );
   DBusPendingCall* pending = wrapper->pending();
   DBusConnection* conn = wrapper->conn();

   // Be sure to have an updated connection status.
   if ( i_dispatch != 0 && i_dispatch->isTrue() )
   {
      dbus_connection_read_write_dispatch( conn, 0 );
   }

   vm->regA().setBoolean( dbus_pending_call_get_completed( pending ) );
}

/*#
   @method cancel DBusPendingCall
   @brief Cancels a pending call.

   Interrupts any wait on this call and notifies the DBUS system (and the other end) that we're
   not interested anymore in the call.
*/
FALCON_FUNC  DBusPendingCall_cancel( VMachine *vm )
{
   Mod::DBusPendingWrapper* wrapper = static_cast<Mod::DBusPendingWrapper*>( vm->self().asObject()->getUserData() );
   DBusPendingCall* pending = wrapper->pending();
   dbus_pending_call_cancel( pending );
}

//============================================================
// Pending call
//

/*#
   @class DBusMessage
   @brief Handle dbus messages

   This class is returned by @a DBus.popMessage and cannot be directly instantiated.
*/

FALCON_FUNC  DBusMessage_getDestination( VMachine *vm )
{
   Mod::DBusMessageWrapper *msgwp = static_cast< Mod::DBusMessageWrapper* >( vm->self().asObject()->getUserData() );
   DBusMessage *msg = msgwp->msg();
   
   const char *destination = dbus_message_get_destination( msg );
   if( destination )
      vm->retval( new CoreString( destination ) );
}

FALCON_FUNC  DBusMessage_getSender( VMachine *vm )
{
   Mod::DBusMessageWrapper *msgwp = static_cast< Mod::DBusMessageWrapper* >( vm->self().asObject()->getUserData() );
   DBusMessage *msg = msgwp->msg();
   
   const char *sender = dbus_message_get_sender( msg );
   if( sender )
      vm->retval( new CoreString( sender ) );
}

FALCON_FUNC  DBusMessage_getPath( VMachine *vm )
{
   Mod::DBusMessageWrapper *msgwp = static_cast< Mod::DBusMessageWrapper* >( vm->self().asObject()->getUserData() );
   DBusMessage *msg = msgwp->msg();
   
   const char *path = dbus_message_get_path( msg );
   if( path )
      vm->retval( new CoreString( path ) );
}

FALCON_FUNC  DBusMessage_getInterface( VMachine *vm )
{
   Mod::DBusMessageWrapper *msgwp = static_cast< Mod::DBusMessageWrapper* >( vm->self().asObject()->getUserData() );
   DBusMessage *msg = msgwp->msg();
   
   const char *interface = dbus_message_get_interface( msg );
   if( interface )
      vm->retval( new CoreString( interface ) );
}

FALCON_FUNC  DBusMessage_getMember( VMachine *vm )
{
   Mod::DBusMessageWrapper *msgwp = static_cast< Mod::DBusMessageWrapper* >( vm->self().asObject()->getUserData() );
   DBusMessage *msg = msgwp->msg();
   
   const char *member = dbus_message_get_member( msg );
   if( member )
      vm->retval( new CoreString( member ) );
}

FALCON_FUNC  DBusMessage_getArgs( VMachine *vm )
{
   DBusMessage *msg = static_cast< Mod::DBusMessageWrapper* >( vm->self().asObject()->getUserData() )->msg();
   Item target;
   if ( s_extract_return( target, msg ) )
      vm->retval( target );
}
   
//======================================================
// DBusError error
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
   @brief Error raised by DBus functions.
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
