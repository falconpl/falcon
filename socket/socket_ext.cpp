/*
   FALCON - The Falcon Programming Language.
   FILE: socket_ext.cpp
   $Id: socket_ext.cpp,v 1.9 2007/08/11 10:22:44 jonnymind Exp $

   Falcon VM interface to socket module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50
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
   Falcon VM interface to socket module.
*/

#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/stream.h>
#include <falcon/memory.h>
#include <errno.h>

#include "socket_sys.h"
#include "socket_ext.h"

namespace Falcon {
namespace Ext {

/**
   Return the machine name
*/

FALCON_FUNC  falcon_getHostName( ::Falcon::VMachine *vm )
{
   String *s = new GarbageString( vm );
   if ( ::Falcon::Sys::getHostName( *s ) )
      vm->retval( s );
   else {
      delete s;
      vm->raiseModError(  new NetError( ErrorParam( 1130, __LINE__ ).
         desc( "Generic network error" ).sysError( (uint32) errno ) ) );
   }
}

FALCON_FUNC  resolveAddress( ::Falcon::VMachine *vm )
{
   Item *address = vm->param( 0 );
   if ( address == 0 || ! address->isString() )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   Sys::Address addr;
   addr.set( *address->asString() );
   if ( ! addr.resolve() ) {
      vm->raiseModError(  new NetError( ErrorParam( 1132, __LINE__ ).
         desc( "System error in resolving address" ).sysError( (uint32) addr.lastError() ) ) );
   }

   CoreArray *ret = new CoreArray( vm, addr.getResolvedCount() );
   String dummy;
   int32 port;

   for ( int i = 0; i < addr.getResolvedCount(); i ++ )
   {
      String *s = new GarbageString( vm );
      addr.getResolvedEntry( i, *s, dummy, port );
      ret->append( s );
   }

   vm->retval( ret );
}

FALCON_FUNC  socketErrorDesc( ::Falcon::VMachine *vm )
{
   Item *code = vm->param( 0 );
   if ( code == 0 || ! code->isInteger() ) {
      vm->retnil();
   }
   else {
      String *str = new GarbageString( vm );
      if ( Sys::getErrorDesc( code->asInteger(), *str ) )
         vm->retval( str );
      else
         vm->retnil();
   }
}


// ==============================================
// Class Socket
// ==============================================

FALCON_FUNC  Socket_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->setProperty( "timedOut", (int64) 0 );
}

FALCON_FUNC  Socket_setTimeout( ::Falcon::VMachine *vm )
{
   // get the address from the parameter.
   Item *i_to = vm->param(0);
   if ( i_to == 0 || ! i_to->isOrdinal() )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   tcps->timeout( (int32) i_to->forceInteger() );
}

FALCON_FUNC  Socket_getTimeout( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   vm->retval( tcps->timeout() );
}

FALCON_FUNC  Socket_dispose( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   tcps->terminate();
   vm->retnil();
}

FALCON_FUNC  Socket_readAvailable( ::Falcon::VMachine *vm )
{
   Item *to = vm->param( 0 );
   if ( to != 0 && ! to->isOrdinal() )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   int64 timeout = to == 0 ? 0 : to->forceInteger();

   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();

   if ( ! tcps->readAvailable( (int32)timeout ) )
   {
      if ( tcps->lastError() == 0 ) {
         self->setProperty( "timedOut", (int64) 1 );
         vm->retval( (int64) 0 );
      }
      else {
         // error
         vm->raiseModError(  new NetError( ErrorParam( 1139, __LINE__ ).
            desc( "Generic socket error" ).sysError( (uint32) tcps->lastError() ) ) );
         self->setProperty( "lastError", tcps->lastError() );
         self->setProperty( "timedOut", (int64) 0 );
      }
   }
   else {
      self->setProperty( "timedOut", (int64) 0 );
      vm->retval( (int64) 1 );
   }
}

FALCON_FUNC  Socket_writeAvailable( ::Falcon::VMachine *vm )
{
   Item *to = vm->param( 0 );
   if ( to != 0 && ! to->isOrdinal() )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   int64 timeout = to == 0 ? 0 : to->forceInteger();

   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();

   if ( ! tcps->writeAvailable( (int32)timeout ) )
   {
      if ( tcps->lastError() == 0 ) {
         self->setProperty( "timedOut", (int64) 1 );
         vm->retval( (int64) 0 );
      }
      else {
         // error
         vm->raiseModError(  new NetError( ErrorParam( 1139, __LINE__ ).
            desc( "Generic socket error" ).sysError( (uint32) tcps->lastError() ) ) );
         self->setProperty( "lastError", tcps->lastError() );
         self->setProperty( "timedOut", (int64) 0 );
      }
   }
   else {
      self->setProperty( "timedOut", (int64) 0 );
      vm->retval( (int64) 1 );
   }
}

FALCON_FUNC  Socket_getHost( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   String *s = new GarbageString( vm );
   if ( tcps->address().getAddress( *s ) ) {
      vm->retval( s );
      return;
   }
   delete s;
   vm->retnil();
}

FALCON_FUNC  Socket_getService( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   String *s = new GarbageString( vm );
   if ( tcps->address().getService( *s ) ) {
      vm->retval( s );
      return;
   }
   delete s;
   vm->retnil();
}

FALCON_FUNC  Socket_getPort( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::TCPSocket *) self->getUserData();
   vm->retval( tcps->address().getPort() );
}

// ==============================================
// Class TCPSocket
// ==============================================

FALCON_FUNC  TCPSocket_init( ::Falcon::VMachine *vm )
{
   Sys::TCPSocket *skt = new Sys::TCPSocket( true );
   CoreObject *self = vm->self().asObject();

   self->setProperty( "timedOut", (int64) 0 );

   self->setUserData( skt );

   if ( skt->lastError() != 0 ) {
      self->setProperty( "lastError", (int64) skt->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( 1131, __LINE__ ).
         desc( "Socket creation failed" ).sysError( (uint32) skt->lastError() ) ) );
   }
}


FALCON_FUNC  TCPSocket_connect( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   // get the address from the parameter.
   Item *i_server = vm->param(0);
   Item *i_service = vm->param(1);
   if ( i_server == 0 || i_service == 0 || ! i_server->isString() ||
         ! i_service->isString() )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, S" ) ) );
      return;
   }

   // try to resolve them.
   Sys::Address addr;
   addr.set( *i_server->asString(), *i_service->asString() );

   //in case of failed resolution, raise an error.
   if ( ! addr.resolve() ) {
      self->setProperty( "lastError", addr.lastError() );
      vm->raiseModError(  new NetError( ErrorParam( 1133, __LINE__ ).
         desc( "resolution failure" ).sysError( (uint32) addr.lastError() ) ) );
      return;
   }

   // connection
   if ( tcps->connect( addr ) ) {
      vm->retval( (int64) 1 );
      self->setProperty( "timedOut", (int64)0 );
      return;
   }

   // connection not complete
   if ( tcps->lastError() == 0 ) {
      // timed out
      self->setProperty( "timedOut", (int64) 1 );
      vm->retval( (int64) 0 );
   }
   else {
      self->setProperty( "lastError", tcps->lastError() );
      self->setProperty( "timedOut", (int64) 0 );
      vm->raiseModError(  new NetError( ErrorParam( 1134, __LINE__ ).
         desc( "Error during connection" ).sysError( (uint32) tcps->lastError() ) ) );
   }
}

FALCON_FUNC  TCPSocket_isConnected( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   if ( ! tcps->isConnected() ) {
      // timed out?
      if ( tcps->lastError() == 0 ) {
         self->setProperty( "timedOut", (int64) 1 );
         vm->retval( (int64) 0 );
         return;
      }

      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( 1134, __LINE__ ).
         desc( "Error during connection" ).sysError( (uint32) tcps->lastError() ) ) );
   }
   else {
      // success
      vm->retval( (int64) 1 );
   }

   self->setProperty( "timedOut", (int64) 0 );
}

FALCON_FUNC  TCPSocket_send( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   Item *data = vm->param( 0 );
   Item *length = vm->param( 1 );
   Item *start = vm->param( 2 );
   if ( data == 0 || ! data->isString() ||
       ( length != 0 && ! length->isOrdinal() ) ||
       ( start != 0 && ! start->isOrdinal() )
      )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, [I], [I]" ) ) );
      return;
   }

   String *dataStr = data->asString();
   int32 start_pos = start == 0 ? 0 : (int32) start->forceInteger();
   if ( start_pos < 0 ) start_pos = 0;
   int32 count = length == 0 ? -1 : (int32) length->forceInteger();

   if ( count < 0 || count + start_pos > (int32) dataStr->size() ) {
      count = dataStr->size() - start_pos;
   }

   int32 res = tcps->send( dataStr->getRawStorage() + start_pos, count );
   if( res == -1 ) {
      self->setProperty( "Error in sending", tcps->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( 1136, __LINE__ ).
         desc( "Error in sending" ).sysError( (uint32) tcps->lastError() ) ) );
      return;
   }
   else if ( res == -2 )
      self->setProperty( "timedOut", (int64) 1 );
   else
      self->setProperty( "timedOut", (int64) 0 );

   vm->retval( (int64) res );
}

FALCON_FUNC  TCPSocket_recv( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   Item *target = vm->param(0);
   String *cs_target;
   // if the third parameter is a not number, the second must be a string;
   // if the string is missing, we must create a new appropriate target.
   Item *last = vm->param(1);


   int32 size;
   bool returnTarget;

   if ( target == 0 ) {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "X, [N]" ) ) );
      return;
   }

   if ( last != 0 ) {
      size = (int32) last->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError(  new ParamError( ErrorParam( e_param_range, __LINE__ ).
            extra( "size less than 0" ) ) );
         return;
      }

      if ( target->isString() )
      {
         cs_target = target->asString();
         // reserve a little size; it is ignored when there's enough space.
         cs_target->reserve( size );
      }
      else {
         vm->raiseModError(  new ParamError( ErrorParam( e_param_type, __LINE__ ).
            extra( "Given a size, the first parameter must be a string" ) ) );
         return;
      }
      returnTarget = false;
   }
   // we have only the second parameter.
   // it MUST be a string or an integer .
   else if ( target->isString() )
   {
      cs_target = target->asString();
      size = cs_target->allocated();

      if ( size <= 0 ) {
         size = cs_target->size();
         if ( size <= 0 ) {
            vm->raiseModError(  new ParamError( ErrorParam( e_param_range, __LINE__ ).
               extra( "Passed String must have space" ) ) );
            return;
         }

         cs_target->reserve( size ); // force to bufferize
      }

      returnTarget = false;
   }
   else if ( target->isInteger() )
   {
      size = (int32) target->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError(  new ParamError( ErrorParam( e_param_range, __LINE__ ).
            extra( "size less than 0" ) ) );
         return;
      }
      cs_target = new GarbageString( vm );
      cs_target->reserve( size );
      // no need to store for garbage, as we'll return this.
      returnTarget = true;
   }
   else
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_param_type, __LINE__ ).
         extra( "X, [S|I]" ) ) );
      return;
   }


   size = tcps->recv( cs_target->getRawStorage(), size );
   if( size == -1 ) {
      self->setProperty( "lastError", tcps->lastError() ) ;
      vm->raiseModError(  new NetError( ErrorParam( 1137, __LINE__ ).
         desc( "Error in receiving" ).sysError( (uint32) tcps->lastError() ) ) );
      return;
   }
   else if ( size == -2 ) {
      self->setProperty( "timedOut", (int64) 1 ) ;
      size = -1;
   }
   else
      self->setProperty( "timedOut", (int64) 0 ) ;

   if ( size > 0 )
      cs_target->size( size );

   if ( returnTarget ) {
      vm->retval( cs_target );
   }
   else {
      vm->retval((int64) size );
   }
}

FALCON_FUNC  TCPSocket_closeRead( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   if ( ! tcps->closeRead() ) {
      // may time out
      if ( tcps->lastError() == 0 ) {
         self->setProperty( "timedOut", (int64) 1 );
         vm->retval( (int64) 0 );
         return;
      }

      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      self->setProperty( "timedOut", (int64) 0 );
      vm->raiseModError(  new NetError( ErrorParam( 1138, __LINE__ ).
         desc( "Error in closing socket" ).sysError( (uint32) tcps->lastError() ) ) );
      return;
   }

   vm->retval( (int64) 1 );
}

FALCON_FUNC  TCPSocket_closeWrite( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   self->setProperty( "timedOut", (int64) 0 );

   if ( tcps->closeWrite() ) {
      vm->retval( (int64) 1 );
   }
   else {
      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( 1138, __LINE__ ).
         desc( "Error in closing socket" ).sysError( (uint32) tcps->lastError() ) ) );
   }
}

FALCON_FUNC  TCPSocket_close( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   if ( ! tcps->close() ) {
      // may time out
      if ( tcps->lastError() == 0 ) {
         self->setProperty( "timedOut", (int64) 1 );
         vm->retval( (int64) 0 );
         return;
      }

      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      self->setProperty( "timedOut", (int64) 0 );
      vm->raiseModError(  new NetError( ErrorParam( 1138, __LINE__ ).
         desc( "Error in closing socket" ).sysError( (uint32) tcps->lastError() ) ) );
      return;
   }

   vm->retval( (int64) 1 );
}

// ==============================================
// Class UDPSocket
// ==============================================

/**
   UDPSocket() --> send only socket
   UDPSocket( ADDRESS, [SERVICE] ) --> named socket
*/

FALCON_FUNC  UDPSocket_init( ::Falcon::VMachine *vm )
{
   Item *address_i = vm->param( 0 );
   Item *service_i = vm->param( 1 );

   Sys::UDPSocket *skt;

   if ( address_i != 0 )
   {
      if ( ! address_i->isString() || ( service_i != 0 && ! service_i->isString() ) )
      {
         vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "S, [S]" ) ) );
         return;
      }
      Sys::Address addr;
      if ( service_i != 0 )
         addr.set( *address_i->asString(), *service_i->asString() );
      else
         addr.set( *address_i->asString() );
      skt = new Sys::UDPSocket( addr );
   }
   else {
      skt = new Sys::UDPSocket();
   }

   CoreObject *self = vm->self().asObject();
   self->setUserData( skt );

   if ( skt->lastError() != 0 ) {
      self->setProperty( "lastError", (int64) skt->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( 1131, __LINE__ ).
         desc( "Socket creation error" ).sysError( (uint32) skt->lastError() ) ) );
   }
}



FALCON_FUNC  UDPSocket_sendTo( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::UDPSocket *udps = (Sys::UDPSocket *) self->getUserData();

   Item *addr = vm->param( 0 );
   Item *srvc = vm->param( 1 );
   Item *data = vm->param( 2 );
   Item *length = vm->param( 3 );
   Item *start = vm->param( 4 );
   if (( addr == 0 || ! addr->isString() ) ||
       ( srvc == 0 || ! srvc->isString() ) ||
       ( data == 0 || ! data->isString() ) ||
       ( length != 0 && ! length->isOrdinal() ) ||
       ( start != 0 && ! start->isOrdinal() )
      )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, S, [N], [N]" ) ) );
      return;
   }

   String *dataStr = data->asString();
   int32 start_pos = start == 0 ? 0 : (int32) start->forceInteger();
   if ( start_pos < 0 ) start_pos = 0;
   int32 count = length == 0 ? -1 : (int32) length->forceInteger();

   if ( count < 0 || count + start_pos > (int32) dataStr->size() ) {
      count = dataStr->size() - start_pos;
   }

   Sys::Address target;
   target.set( *addr->asString(), *srvc->asString() );
   int32 res = udps->sendTo( dataStr->getRawStorage() + start_pos, count, target );
   if( res == -1 ) {
      self->setProperty( "lastError", udps->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( 1136, __LINE__ ).
         desc( "Error in sending" ).sysError( (uint32) udps->lastError() ) ) );
      return;
   }
   else if ( res == -2 )
      self->setProperty( "timedOut", (int64) 1 );
   else
      self->setProperty( "timedOut", (int64) 0 );

   vm->retval( (int64) res );
}

FALCON_FUNC  UDPSocket_recv( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::UDPSocket *udps = (Sys::UDPSocket *) self->getUserData();

   Item *target = vm->param(0);
   String *cs_target;
   // if the third parameter is a not number, the second must be a string;
   // if the string is missing, we must create a new appropriate target.
   Item *last = vm->param(1);

   int32 size;
   bool returnTarget;

   if ( target == 0 ) {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "X, [N]" ) ) );
      return;
   }

   if ( last != 0 ) {
      size = (int32) last->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError(  new ParamError( ErrorParam( e_param_range, __LINE__ ).
            extra( "size less than 0" ) ) );
         return;
      }

      if ( target->isString() )
      {
         cs_target = target->asString();
         // reserve a little size; it is ignored when there's enough space.
         cs_target->reserve( size );
      }
      else {
         vm->raiseModError(  new ParamError( ErrorParam( e_param_type, __LINE__ ).
            extra( "Given a size, the first parameter must be a string" ) ) );
         return;
      }
      returnTarget = false;
   }
   // we have only the second parameter.
   // it MUST be a string or an integer .
   else if ( target->isString() )
   {
      cs_target = target->asString();
      size = cs_target->size();

      if ( size <= 0 ) {
         vm->raiseModError(  new ParamError( ErrorParam( e_param_range, __LINE__ ).
            extra( "Passed String must have space" ) ) );
         return;
      }

      cs_target->reserve( size ); // force to bufferize
      returnTarget = false;
   }
   else if ( target->isInteger() )
   {
      size = (int32) target->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError( new  ParamError( ErrorParam( e_param_range, __LINE__ ).
            extra( "size less than 0" ) ) );
         return;
      }
      cs_target = new GarbageString( vm );
      cs_target->reserve( size );
      // no need to store for garbage, as we'll return this.
      returnTarget = true;
   }
   else
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_param_type, __LINE__ ).
         extra( "X, S|I" ) ) );
      return;
   }

   Sys::Address from;
   size = udps->recvFrom( cs_target->getRawStorage(), size, from );
   if( size == -1 ) {
      self->setProperty( "lastError", udps->lastError() ) ;
      vm->raiseModError(  new  NetError( ErrorParam( 1137, __LINE__ ).
         desc( "Error in sending" ).sysError( (uint32) udps->lastError() ) ) );
      return;
   }
   else if ( size == -2 ) {
      self->setProperty( "timedOut", (int64) 1 ) ;
      size = -1;
   }
   else {
      self->setProperty( "timedOut", (int64) 0 ) ;
      String temp;
      from.getAddress( temp );
      self->setProperty( "remote", temp );
      from.getService( temp );
      self->setProperty( "remoteService", temp );
   }

   if ( size > 0 )
      cs_target->size( size );

   if ( returnTarget ) {
      vm->retval( cs_target );
   }
   else {
      vm->retval((int64) size );
   }
}

FALCON_FUNC  UDPSocket_broadcast( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::UDPSocket *udps = (Sys::UDPSocket *) self->getUserData();

   udps->turnBroadcast( true );
}


// ==============================================
// Class Server socket
// ==============================================

FALCON_FUNC  TCPServer_init( ::Falcon::VMachine *vm )
{
   Sys::ServerSocket *skt = new Sys::ServerSocket( true );
   CoreObject *self = vm->self().asObject();

   self->setUserData( skt );

   if ( skt->lastError() != 0 ) {
      self->setProperty( "lastError", (int64) skt->lastError() );
      vm->raiseModError(  new  NetError( ErrorParam( 1139, __LINE__ ).
         desc( "Server Socket creation" ).sysError( (uint32) skt->lastError() ) ) );
   }
}

FALCON_FUNC  TCPServer_dispose( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ServerSocket *tcps = (Sys::ServerSocket *) self->getUserData();
   tcps->terminate();
   vm->retnil();

}

/**
   server.bind( address, service )
   server.bind( service )
*/
FALCON_FUNC  TCPServer_bind( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ServerSocket *srvs = (Sys::ServerSocket *) self->getUserData();

   // paramters
   Item *i_first = vm->param( 0 );
   Item *i_second = vm->param( 1 );

   if ( i_first == 0 || ! i_first->isString() || ( i_second != 0 && ! i_second->isString() ) )
   {
         vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "S, [N]" ) ) );
      return;
   }

   Sys::Address addr;
   if( i_second != 0 )
      addr.set( *i_first->asString(), *i_second->asString() );
   else
      addr.set( "0.0.0.0", *i_first->asString() );

   if ( ! srvs->bind( addr ) )
   {
      self->setProperty( "lastError", srvs->lastError() );
      vm->raiseModError(  new  NetError( ErrorParam( 1140, __LINE__ ).
         desc( "Can't bind socket to address" ).sysError( (uint32) srvs->lastError() ) ) );
   }

   vm->retnil();
}

/**
   server.accept( [timeout] ) --> socket/nil
*/
FALCON_FUNC  TCPServer_accept( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ServerSocket *srvs = (Sys::ServerSocket *) self->getUserData();

   // timeout as first parameter.
   Item *to = vm->param( 0 );

   if( to == 0 ) {
      srvs->timeout( -1 );
   }
   else if ( to->isOrdinal() ) {
      srvs->timeout( (int32) to->forceInteger() );
   }
   else {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   Sys::TCPSocket *skt = srvs->accept();
   if ( srvs->lastError() != 0 )
   {
      self->setProperty( "lastError", srvs->lastError() );
      vm->raiseModError(  new  NetError( ErrorParam( 1141, __LINE__ ).
         desc( "Error while accepting connections" ).sysError( (uint32) srvs->lastError() ) ) );
      return;
   }

   if ( skt == 0 ) {
      vm->retnil();
      return;
   }

   Item *tcp_class = vm->findGlobalItem( "TCPSocket" );
   fassert( tcp_class != 0 );
   CoreObject *ret_s = tcp_class->asClass()->createInstance();
   ret_s->setUserData( skt );

   vm->retval( ret_s );
}

FALCON_FUNC  NetError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new NetError ) );

   ::Falcon::core::Error_init( vm );
}

}
}

/* end of socket_ext.cpp */
