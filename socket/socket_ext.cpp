/*
   FALCON - The Falcon Programming Language.
   FILE: socket_ext.cpp

   Falcon VM interface to socket module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004-2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
#include "socket_st.h"

/*#
   @beginmodule feather_socket
*/

namespace Falcon {
namespace Ext {

/*#
   @function getHostName
   @brief Retreives the host name of the local machine.
   @return A string containing the local machine name.
   @raise @a NetError if the host name can't be determined.

   Returns the network name under which the machine is known to itself. By
   calling @a resolveAddress on this host name, it is possible to determine all
   the addressess of the interfaces that are available for networking.

   If the system cannot provide an host name for the local machine, a NetError
   is raised.

   @note If the function fails, it is possible to retreive local addresses using
      through @a resolveAddress using the the name "localhost".
*/

FALCON_FUNC  falcon_getHostName( ::Falcon::VMachine *vm )
{
   String *s = new GarbageString( vm );
   if ( ::Falcon::Sys::getHostName( *s ) )
      vm->retval( s );
   else {
      delete s;
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_GENERIC, __LINE__ )
         .desc( FAL_STR( sk_msg_generic ) )
         .sysError( (uint32) errno ) ) );
   }
}

/*#
   @function resolveAddress
   @brief Resolve a network address in a list of numeric IP fields.
   @param address An host name, quad dot IP address or IPV6 address.
   @return An array of IPv4 or IPv6 addresses.
   @raise @a NetError if the name resolution service is not available.

   This function tries to resolve an address or an host name into a list of
   addresses that can be used to connect directly via the protocols that are
   available on the machine. The way in which the function can resolve the
   addresses depends on the string that has been given, on the underlying system
   and on the name resolution services that the system can access. Also, the time
   by which a positive or negative result can be returned varies greatly between
   different systems. The caller must consider this function as potentially
   blocking the VM for a long time.

   The other members of the Socket module do not need to be provided with already
   resolved addresses. In example, the connect method of the TCPSocket class can be
   provided with a host name or with an IP address; in the first case, the connect
   method will resolve the target address on its own.

   This function can be used on the value returned by @a getHostName, or using
   "localhost" as the @a address parameter, to receive a
   list of the interfaces that are available under the network name of the host the
   VM is running on. This is useful to i.e. bind some services only to one of the
   available interfaces.

   The values in the returned array are the ones provided by the underlying name
   resolution system. They are not necessarily unique, and their order may change
   across different calls.

   If the function cannot find any host with the given name, an empty array is
   returned; if the name resolution service is not accessible, or if accessing it
   causes a system error, the function will raise a @a NetError.
*/
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
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_RESOLV, __LINE__ )
         .desc( FAL_STR( sk_msg_errresolv ) )
         .sysError( (uint32) addr.lastError() ) ) );
      return;
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

/*#
   @function socketErrorDesc
   @brief Describe the meaning of a networking related system error code.
   @param code The error code to be described.
   @return A string with a textual description of the error.

   This function invokes the system code-to-meaning translation for
   networking errors. The language in which the error code is described
   depends on the underlying system,

   The function can be applied to the fsError field of the @a NetError class
   to get a descriptive reason of why some operation failed, or on @a Socket.lastError.
*/
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

/*#
   @class Socket
   @brief TCP/IP networking base class,
   The Socket class is the base class for both UDP and TCP socket.
   It provides common methods and properties,
   and so it should not be directly instantiated.

   @prop timedout True if the last operation has timed out. See @a Socket,setTimeout..

   @prop lastError Numeric value of system level error that has occoured on the socket.
      @a getErrorDescription may be used to get a human-readable description of the error.
      The error is usually also written in the fsError field of the exceptions,
      if case they are caught.
*/


FALCON_FUNC  Socket_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->getProperty( "timedOut")->setBoolean( false );
}

/*#
   @method setTimeout Socket
   @param timeout Timeout in seconds and fractions.

   This function sets a default timeout for the read/write operations, or for other
   lengthy operations as connect. If -1 is set (the default at socket creation),
   blocking operation will wait forever, until some data is ready. In this case,
   readAvailable and writeAvailable methods can be used to peek for incoming data.

   If 0 is set, read/write operations will never block, returning immediately if
   data is not available. If a value greater than zero is set, blocking functions
   will wait the specified amount of seconds for their action to complete.

   Whenever an operation times out, the @a Socket,timedout member property
   is set to 1. This allows to distinguish between faulty operations and
   timed out ones.

   @a Socket.readAvailable and @a Socket,writeAvailable methods do not use
   this setting.
*/

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


/*#
   @method getTimeout Socket
   @brief Returns the default timeout for this socket.
   @return A numeric value (seconds or seconds fractions).

   @see Socket.setTimeout
*/

FALCON_FUNC  Socket_getTimeout( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   vm->retval( tcps->timeout() );
}

/*#
   @method dispose Socket
   @brief Closes a socket and frees system resources associated with it.

   Sockets are automatically disposed by garbage collector, but the
   calling program may find useful to free them immediately i.e. in
   tight loops while accepting and serving sockets.
*/

FALCON_FUNC  Socket_dispose( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   tcps->terminate();
   vm->retnil();
}


/*#
   @method readAvailable Socket
   @brief Checks if there is available data to be read on this socket.
   @optparam timeout Wait for a specified time in seconds or fractions.
   @return True if the next read operation would not block.
   @raise InterruptedError In case of asynchronous interruption.

   This method can be used to wait for incoming data on the socket.
   If there are some data available for immediate read, the function returns
   immediately true, otherwise it returns false.

   If @b timeout is not given, the function will return immediately,
   peeking for current read availability status. If it is given, the function
   will wait for a specified amount of seconds, or for some data to become
   available for read, whichever comes first.

   If the timeout value is negative, the function will wait forever,
   until some data is available.

   This wait will block the VM and all the coroutines.

   @note On Unix, this function respects the VirtualMachine interruption
   protocol, and can be asynchronously interrupted from other threads. This
   functionality is not implemented on MS-Windows systems yet, and will be
   provided in version 0.8.12.

   @see Socket.writeAvailable
*/

FALCON_FUNC  Socket_readAvailable( ::Falcon::VMachine *vm )
{
   Item *to = vm->param( 0 );
   if ( to != 0 && ! to->isOrdinal() )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   int64 timeout = to == 0 ? -1 : int64(to->forceNumeric() * 1000.0);

   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   int res;

   if ( (res = tcps->readAvailable( (int32)timeout, &vm->systemData() ) ) <= 0 )
   {
      // interrupted?
      if ( res == -2 )
      {
         vm->interrupted( true, true, true );
         return;
      }

      if ( tcps->lastError() == 0 ) {
         self->getProperty( "timedOut" )->setBoolean(  true  );
         vm->regA().setBoolean( false );
      }
      else {
         vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_GENERIC, __LINE__ )
            .desc( FAL_STR( sk_msg_generic ) )
            .sysError( (uint32) tcps->lastError() ) ) );
         self->setProperty( "lastError", tcps->lastError() );
         self->getProperty( "timedOut" )->setBoolean( false );
      }
   }
   else {
      self->getProperty( "timedOut")->setBoolean(  false  );
      vm->regA().setBoolean( true );
   }
}


/*#
   @method writeAvailable Socket
   @brief Waits for the socket to be ready to write data.
   @optparam timeout Optional wait in seconds.
   @return True if the socket is writeable or becomes writeable during the wait.
   @raise InterruptedError in case of asynchronous interruption.

   This method checks for the socket to be ready for immediate writing. A socket
   may not be ready for writing if the OS system stack is full, which usually means
   that the other side has not been able to forward the received messages to the
   listening application.

   The method will return true in case the socket is available for write, false
   otherwise.

   An optional @b timeout may be specified; in this case, the function will return true
   if the socket is immediately available or if it becomse available before the
   wait expires, false otherwise. The wait blocks the VM, preventing other
   coroutines to be processed as well.

   This function does not take into consideration overall timeout set by
   @a Socket.setTimeout.

   @note On Unix, this function respects the VirtualMachine interruption
   protocol, and can be asynchronously interrupted from other threads. This
   functionality is not implemented on MS-Windows systems yet, and will be
   provided in version 0.8.12.

   @see Socket.readAvailable

*/

FALCON_FUNC  Socket_writeAvailable( ::Falcon::VMachine *vm )
{
   Item *to = vm->param( 0 );
   if ( to != 0 && ! to->isOrdinal() )
   {
      vm->raiseModError(  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "[N]" ) ) );
      return;
   }

   int64 timeout = to == 0 ? -1 : int64(to->forceNumeric() * 1000.0);

   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::Socket *) self->getUserData();
   int res;

   if ( ( res = tcps->writeAvailable( (int32)timeout, &vm->systemData() ) ) <= 0 )
   {
      // interrupted?
      if ( res == -2 )
      {
         vm->interrupted( true, true, true );
         return;
      }

      if ( tcps->lastError() == 0 ) {
         self->getProperty( "timedOut")->setBoolean(  true  );
         vm->regA().setBoolean( false );
      }
      else {
         // error
         vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_GENERIC, __LINE__ )
            .desc( FAL_STR( sk_msg_generic ) )
            .sysError( (uint32) tcps->lastError() ) ) );
         self->setProperty( "lastError", tcps->lastError() );
         self->getProperty( "timedOut")->setBoolean(  false  );
      }
   }
   else {
      self->getProperty( "timedOut" )->setBoolean(  false  );
      vm->regA().setBoolean( true );
   }
}

/*#
   @method getHost Socket
   @brief Gets the host associated with this socket.
   @return The host address.

   For TCP sockets, this method will always return the address of the remote host,
   while in UDP sockets it will be the local address where the socket has been bound.
*/

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

/*#
   @method getService Socket
   @brief Returns the service name (port description) associated with this socket
   @return A string containing the service name.

   For TCP sockets, returns the name of the service to which the socket is connected.
   For UDP sockets, returns the local service from which the messages
   sent through this socket are generated, if an explicit bound request
   has been issued. Returned values are a system-specific 1:1 mapping of numeric
   ports to service names. If the port has not an associated service name,
   the port number is returned as a string value (I.e. port 80 is
   returned as the string “80”).
*/

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


/*#
   @method getPort Socket
   @brief Gets the port associated with this socket.
   @return The port address.

   For TCP sockets, returns a numeric representation of the port to which the socket
   is connected. For UDP sockets, returns the local port from which the messages
   sent through this socket are generated, if an explicit bound request has been issued.
*/

FALCON_FUNC  Socket_getPort( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::Socket *tcps = (Sys::TCPSocket *) self->getUserData();
   vm->retval( tcps->address().getPort() );
}

// ==============================================
// Class TCPSocket
// ==============================================

/*#
   @class TCPSocket
   @from Socket
   @brief Provides full TCP connectivity.

   This class is derived from the @a Socket class, but it also provides methods that can be
   used to open connection towards remote hosts.
*/

/*#
   @init TCPSocket
   @brief Allocate system resources for TCP/IP Connectivity.
   @raise NetError on underlying network error.

   The constructor reserves system resources for the socket.
   If the needed system resources are not available, a NetError is Raised.
*/

FALCON_FUNC  TCPSocket_init( ::Falcon::VMachine *vm )
{
   Sys::TCPSocket *skt = new Sys::TCPSocket( true );
   CoreObject *self = vm->self().asObject();

   self->getProperty( "timedOut" )->setBoolean( false );

   self->setUserData( skt );

   if ( skt->lastError() != 0 ) {
      self->setProperty( "lastError", (int64) skt->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_CREATE, __LINE__ )
         .desc( FAL_STR( sk_msg_errcreate ) )
         .sysError( (uint32) skt->lastError() ) ) );
   }
}


/*#
   @method connect TCPSocket
   @brief Connects this socket to a remote host.
   @param host Remote host to be connected to.
   @param service Port or service name to be connected to.
   @return False if timed out, true if succesful
   @raise NetError in case of underlying connection error during the closing phase.

   Connects with a remote listening TCP host. The operation may fail for a
   system error, in which case a NetError is raised.

   The connection attempt may timeout if it takes more than the time specified
   in @a Socket.setTimeout method. In that case, the @a TCPSocket.isConnected method may check if the
   connection has been established at a later moment. So, it is possible to set the
   socket timeout to 0, and then check periodically for connection success without
   never blocking the VM.

   The host name may be a name to be resolved by the system resoluter or it may
   be an already resolved dot-quad IP, or it may be an IPV6 address.

   @see Socket.setTimeout
*/
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
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_RESOLV, __LINE__ )
         .desc( FAL_STR( sk_msg_errcreate ) )
         .sysError( (uint32) addr.lastError() ) ) );
      return;
   }

   // connection
   if ( tcps->connect( addr ) ) {
      vm->regA().setBoolean( true );
      self->getProperty( "timedOut" )->setBoolean(  false  );
      return;
   }

   // connection not complete
   if ( tcps->lastError() == 0 ) {
      // timed out
      self->getProperty( "timedOut")->setBoolean( true );
      vm->regA().setBoolean( false );
   }
   else {
      self->setProperty( "lastError", tcps->lastError() );
      self->getProperty( "timedOut" )->setBoolean( false );

      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_CONNECT, __LINE__ )
         .desc( FAL_STR( sk_msg_errconnect ) )
         .sysError( (uint32) tcps->lastError() ) ) );
   }
}

/*#
   @method isConnected TCPSocket
   @brief Check if this TCPSocket is currently connected with a remote host.
   @return True if the socket is currently connected, false otherwise.

   This method checks if this TCPSocket is currently connected with a remote host.

   @see TCPSocket.connect
*/
FALCON_FUNC  TCPSocket_isConnected( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   if ( ! tcps->isConnected() ) {
      // timed out?
      if ( tcps->lastError() == 0 ) {
         self->getProperty( "timedOut" )->setBoolean( true );
         vm->regA().setBoolean( false );
         return;
      }

      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      vm->raiseModError( new NetError( ErrorParam( FALSOCK_ERR_CONNECT, __LINE__ )
         .desc( FAL_STR( sk_msg_errconnect ) )
         .sysError( (uint32) tcps->lastError() ) ) );
   }
   else {
      // success
      vm->regA().setBoolean( true );
   }

   self->getProperty( "timedOut" )->setBoolean( false );
}


/*#
   @method send TCPSocket
   @brief Send data on the network connection.
   @param buffer The buffer containing the data to be sent.
   @optparam size Amount of bytes to be sent.
   @optparam start Begin position in the buffer (in bytes).
   @return Number of bytes actually sent through the network layer.
   @raise NetError on network error,

   The @b buffer may be a byte-only string or a
   byte-wide MemBuf; it is possible to send also multibyte strings (i.e. strings
   containing international characters) or multi-byte memory buffers, but in that
   case the sent data may get corrupted as a transmission may deliver only part
   of a character or of a number stored in a memory buffer.

   If a @a size parameter is not specified, the method will try to send the whole
   content of the buffer, otherwise it will send at maximum size bytes. If a
   @a start parameter is specified, then the data sent will be taken starting
   from that position in the buffer (counting in bytes from the start).

   This is useful when sending big buffers in several steps, so that
   it is not necessary to create substrings for each send, sparing both
   CPU and memory.

   The returned value may be 0 in case of timeout, otherwise it will be a
   number between 1 and the requested size. Programs should never assume
   that a succesful @b send has sent all the data.

   In case of error, a @a NetError is raised.

   @see Socket.setTimeout
*/
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
      self->setProperty( "lastError", tcps->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_SEND, __LINE__ )
         .desc( FAL_STR( sk_msg_errsend ) )
         .sysError( (uint32) tcps->lastError() ) ) );
      return;
   }
   else if ( res == -2 )
      self->getProperty( "timedOut" )->setBoolean( true );
   else
      self->getProperty( "timedOut" )->setBoolean( false );

   vm->retval( (int64) res );
}

/*#
   @method recv TCPSocket
   @brief Reads incoming data.
   @param bufOrSize A pre-allocated buffer to fill, or a maximum size in bytes to be read.
   @optParam size Maximum size in bytes to be read.
   @return If @b bufOrSize is a size, returns a filled string buffer, otherwise it returns
      the amount of bytes actually read.
   @raise NetError on network error.

   When the @b bufOrSize parameter is a buffer to be filled
   (i.e. a @b MemBuf or a string created with @b strBuffer), the buffer may be repeatedly used,
   sparing memory and allocation time. If the size parameter is not provided,
   the maximum amount of bytes that can be stored in the buffer will be used as read size limit,

   This size won't change between call, even if the actual length of the data stored in the
   buffer changes.

   If the size parameter is provided, it is used to define the maximum possible
   size of data stored in the buffer. If it is not great enough to store the data,
   it is resized.

   When a target buffer is specified in the @b bufOrSize parameter, the method returns
   number of bytes that has been
   actually read. The size may be zero if the opposite side closed the connection,
   or if the timeout has elapsed. The returned
   size will presumably be smaller than the maximum, and it will be at maximum the
   size of a TCP packet that can travel on the current connection.

   When a maximum size is specified in the @b bufOrSize parameter, the method
   returns a newly allocated buffer on success, and nil on failure.

   In case of system error, a NetError is raised.

   @see Socket.setTimeout
*/
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
            extra( FAL_STR( sk_msg_lesszero) ) ) );
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
            extra( FAL_STR( sk_msg_firstnostring ) ) ) );
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
               extra( FAL_STR( sk_msg_stringnospace ) ) ) );
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
            extra( FAL_STR( sk_msg_lesszero) ) ) );
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
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_RECV, __LINE__ )
         .desc( FAL_STR( sk_msg_errrecv ) )
         .sysError( (uint32) tcps->lastError() ) ) );
      return;
   }
   else if ( size == -2 ) {
      self->getProperty( "timedOut" )->setBoolean(  true  ) ;
      size = -1;
   }
   else
      self->getProperty( "timedOut" )->setBoolean(  false  ) ;

   if ( size > 0 )
      cs_target->size( size );

   if ( returnTarget ) {
      vm->retval( cs_target );
   }
   else {
      vm->retval((int64) size );
   }
}

/*#
   @method closeRead TCPSocket
   @brief Closes a socket read side.
   @return False if timed out, true if succesful
   @raise NetError in case of underlying connection error during the closing phase.

   Closes the socket read side, discarding incominig messages and notifying
   the remote side about the event. The close message must be acknowledged
   by the remote host, so the function may actually fail,
   block and/or timeout.

   After the call, the socket can still be used to write (i.e. to finish writing
   pending data). This informs the remote side we're not going to read anymore,
   and so if the application on the remote host tries to write,
   it will receive an error.

   In case of error, a NetError is raised, while in case of timeout @b false is returned.
   On successful completion, true is returned.

   @see Socket.setTimeout
*/
FALCON_FUNC  TCPSocket_closeRead( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   if ( ! tcps->closeRead() ) {
      // may time out
      if ( tcps->lastError() == 0 ) {
         self->getProperty( "timedOut" )->setBoolean(  true  );
         vm->regA().setBoolean( false );
         return;
      }

      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      self->getProperty( "timedOut" )->setBoolean(  false  );
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_CLOSE, __LINE__ )
         .desc( FAL_STR( sk_msg_errclose ) )
         .sysError( (uint32) tcps->lastError() ) ) );
      return;
   }

   vm->regA().setBoolean( true );
}


/*#
   @method closeWrite TCPSocket
   @brief Closes a socket write side.
   @return False if timed out, true if succesful
   @raise NetError in case of underlying connection error during the closing phase.

   Closes the socket write side, discarding incoming messages and notifying the
   remote side about the event. The close message must be acknowledged by the
   remote host, so the function may actually fail, block and/or timeout.

   After the call, the socket can still be used to read (i.e. to finish reading
   informations incoming from the remote host). This informs the remote side we're
   not going to write anymore, and so if the application on the remote host tries
   to read, it will receive an error.

   In case of error, a NetError is raised, while in case of timeout @b false is returned.
   On successful completion, true is returned.

   @see Socket.setTimeout
*/
FALCON_FUNC  TCPSocket_closeWrite( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   self->getProperty( "timedOut" )->setBoolean(  false  );

   if ( tcps->closeWrite() ) {
      vm->regA().setBoolean( true );
   }
   else {
      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_CLOSE, __LINE__ )
         .desc( FAL_STR( sk_msg_errclose ) )
         .sysError( (uint32) tcps->lastError() ) ) );
   }
}


/*#
   @method close TCPSocket
   @brief Closes the socket.
   @return False if timed out, true if succesful
   @raise NetError in case of underlying connection error during the closing phase.

   Closes the socket, discarding incoming messages and notifying the remote side
   about the event. The close message must be acknowledged by the remote host, so
   the function may actually fail, block and/or timeout - see setTimeout() .

   In case of error, a NetError is raised, while in case of timeout @b false is returned.
   On successful completion @b true is returned.

   @see Socket.setTimeout
*/
FALCON_FUNC  TCPSocket_close( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   if ( ! tcps->close() ) {
      // may time out
      if ( tcps->lastError() == 0 ) {
         self->getProperty( "timedOut" )->setBoolean(  true  );
         vm->regA().setBoolean( false );
         return;
      }

      // an error!
      self->setProperty( "lastError", tcps->lastError() );
      self->getProperty( "timedOut" )->setBoolean(  false  );
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_CLOSE, __LINE__ )
         .desc( FAL_STR( sk_msg_errclose ) )
         .sysError( (uint32) tcps->lastError() ) ) );
      return;
   }

   vm->regA().setBoolean( true );
}

// ==============================================
// Class UDPSocket
// ==============================================

/*#
   @class UDPSocket
   @brief UDP (datagram) manager.
   @from Socket
   @param addrOrService Address at which this server will be listening.
   @optparam service If an address is given, service or port number (as a string) where to listen.
   @raise NetError on system error.

   The UDPSocket class provides support for UDP transmissions (datagrams).

   The constructor reserves the needed system resources and return an
   UDPSocket object that can be used to send and receive datagrams.

   @prop remote Contains the origin address of the last datagram received with the @a UDPSocket.recv method.
   @prop remoteService Contains the origin port of the last datagram received with the @a UDPSocket.recv method.
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
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_CREATE, __LINE__ )
         .desc( FAL_STR( sk_msg_errcreate ) )
         .sysError( (uint32) skt->lastError() ) ) );
   }
}


/*#
   @method sendTo UDPSocket
   @param host Remote host where to send the datagram.
   @param service Remote service or port number where to send the datagram.
   @param buffer The buffer to be sent.
   @optparam size Amount of bytes from the buffer to be sent.
   @optparam start Begin position in the buffer.
   @raise NetError on network error.

   This method works as the TCPSocket.send method, with the
   main difference that the outgoing datagram can be directed towards a
   specified @b host, and that a whole datagram is always completely
   filled before being sent, provided that the specified size
   does not exceed datagram size limits.

   The @a host parameter may be an host name to be resolved or an address;
   if the @a UDPSocket.broadcast method has been successfully called,
   it may be also a multicast or broadcast address.

   The @a service parameter is a string containing either a service name
   (i.e. "http") or  a numeric port number (i.e. "80", as a string).

   The @b buffer may be a byte-only string or a
   byte-wide MemBuf; it is possible to send also multibyte strings (i.e. strings
   containing international characters) or multi-byte memory buffers, but in that
   case the sent data may get corrupted as a transmission may deliver only part
   of a character or of a number stored in a memory buffer.

   If a @b size parameter is not specified, the method will try to send the whole
   content of the buffer, otherwise it will send at maximum size bytes. If a
   @b start parameter is specified, then the data sent will be taken starting
   from that position in the buffer (counting in bytes from the start).

   This is useful when sending big buffers in several steps, so that
   it is not necessary to create substrings for each send, sparing both
   CPU and memory.

   The returned value may be 0 in case of timeout, otherwise it will be a
   number between 1 and the requested size. Programs should never assume
   that a succesful @b sendTo has sent all the data.

   In case of error, a @a NetError is raised.

   @see Socket.setTimeout
*/
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
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_SEND, __LINE__ )
         .desc( FAL_STR( sk_msg_errsend ) )
         .sysError( (uint32) udps->lastError() ) ) );
      return;
   }
   else if ( res == -2 )
      self->getProperty( "timedOut" )->setBoolean( true );
   else
      self->getProperty( "timedOut" )->setBoolean( false );

   vm->retval( (int64) res );
}


/*#
   @method recv UDPSocket
   @brief Reads incoming data.
   @param bufOrSize A pre-allocated buffer to fill, or a maximum size in bytes to be read.
   @optParam size Maximum size in bytes to be read.
   @return If @b bufOrSize is a size, returns a filled string buffer, otherwise it returns
      the amount of bytes actually read.
   @raise NetError on network error.

   This method works as the @a TCPSocket.recv method, with the only
   difference that the incoming datagram is always completely read,
   provided that the specified size is enough to store the data.

   Also, the @a UDPSocket.remote and @a UDPSocket.remoteService properties
   of the receiving object are filled with the address and port of the host
   sending the packet.

   When the @b bufOrSize parameter is a buffer to be filled
   (i.e. a @b MemBuf or a string created with @b strBuffer), the buffer may be repeatedly used,
   sparing memory and allocation time. If the size parameter is not provided,
   the maximum amount of bytes that can be stored in the buffer will be used as read size limit,

   This size won't change between call, even if the actual length of the data stored in the
   buffer changes.

   If the size parameter is provided, it is used to define the maximum possible
   size of data stored in the buffer. If it is not great enough to store the data,
   it is resized.

   When a target buffer is specified in the @b bufOrSize parameter, the method returns
   number of bytes that has been
   actually read. The size may be zero if the opposite side closed the connection,
   or if the timeout has elapsed. The returned
   size will presumably be smaller than the maximum, and it will be at maximum the
   size of a TCP packet that can travel on the current connection.

   When a maximum size is specified in the @b bufOrSize parameter, the method
   returns a newly allocated buffer on success, and nil on failure.

   In case of system error, a NetError is raised.

   @see Socket.setTimeout
*/
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
            extra( FAL_STR( sk_msg_lesszero) ) ) );
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
            extra( FAL_STR( sk_msg_firstnostring ) ) ) );
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
            extra( FAL_STR( sk_msg_stringnospace ) ) ) );
         return;
      }

      cs_target->reserve( size ); // force to bufferize
      returnTarget = false;
   }
   else if ( target->isInteger() )
   {
      size = (int32) target->forceInteger();
      if ( size <= 0 ) {
         vm->raiseModError(  new ParamError( ErrorParam( e_param_range, __LINE__ ).
            extra( FAL_STR( sk_msg_lesszero) ) ) );
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
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_SEND, __LINE__ )
         .desc( FAL_STR( sk_msg_errsend ) )
         .sysError( (uint32) udps->lastError() ) ) );
      return;
   }
   else if ( size == -2 ) {
      self->getProperty( "timedOut" )->setBoolean( true ) ;
      size = -1;
   }
   else {
      self->getProperty( "timedOut" )->setBoolean( false ) ;
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


/*#
   @method broadcast UDPSocket
   @brief Activates broadcasting and multicasting abilities on this UDP socket.
   @raise NetError on system error.

   This is provided as a method separated from the socket constructor as,
   on some systems, this call  requires administrator privileges to be successful.
*/
FALCON_FUNC  UDPSocket_broadcast( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::UDPSocket *udps = (Sys::UDPSocket *) self->getUserData();

   udps->turnBroadcast( true );
}


// ==============================================
// Class Server socket
// ==============================================

/*#
   @class TCPServer
   @brief Encapsulates a TCP network service provider.

   This class is actually a factory of TCPSockets, that are created as incoming
   connections are received. As such, it is not derived from the Socket class.

   @prop lastError Numeric value of system level error that has occoured on the socket.
      @a socketErrorDesc may be used to get a human-readable description of the error.
      The error is usually also written in the fsError field of the exceptions,
      if case they are caught.
*/
/*#
   @init TCPServer
   @brief Sets up the server.
   @raise NetError on system error.

   The constructor reserves system resources needed to create
   sockets and return a TPCServer object that can be used to
   accept incoming TCP connections.

   If the needed system resources are not available, a NetError is raised.
*/

FALCON_FUNC  TCPServer_init( ::Falcon::VMachine *vm )
{
   Sys::ServerSocket *skt = new Sys::ServerSocket( true );
   CoreObject *self = vm->self().asObject();

   self->setUserData( skt );

   if ( skt->lastError() != 0 ) {
      self->setProperty( "lastError", (int64) skt->lastError() );
      vm->raiseModError(  new NetError( ErrorParam( FALSOCK_ERR_CREATE, __LINE__ )
         .desc( FAL_STR( sk_msg_errcreate ) )
         .sysError( (uint32) skt->lastError() ) ) );
   }
}

/*#
   @method dispose TCPServer
   @brief Closes the service and disposes the resources,

   Collects immediately this object and frees the related
   system resources.
   Using this object after this call causes undefined results.
*/

FALCON_FUNC  TCPServer_dispose( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ServerSocket *tcps = (Sys::ServerSocket *) self->getUserData();
   tcps->terminate();
   vm->retnil();

}

/*#
   @method bind TCPServer
   @brief Specify the address and port at which this server will be listening.
   @param addrOrService Address at which this server will be listening.
   @optparam service If an address is given, service or port number (as a string) where to listen.
   @raise NetError on system error.

   This method binds the server port and start listening for incoming connections.
   If passing two parameters, the first one is considered to be one of the address
   that are available on local interfaces; the second one is the port or service
   name where the server will open a listening port.

   If an address is not provided, that is, if only one parameter is passed,
   the server will listen on all the local interfaces. It is possible to
   specify jolly IPv4 or IPv6 addresses (i.e. "0.0.0.0") to listen on
   all the interfaces.

   In case the system cannot bind the required address, a NetError is raised.
   After a succesful @b bind call, @a TCPServer.accept may be called to create TCPSocket that
   can serve incoming connections.
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
            extra( "S, [S]" ) ) );
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
      vm->raiseModError(  new  NetError( ErrorParam( FALSOCK_ERR_BIND, __LINE__ )
         .desc( FAL_STR(sk_msg_errbind) )
         .sysError( (uint32) srvs->lastError() ) ) );
   }

   vm->retnil();
}

/*#
   @method accept TCPServer
   @brief Waits for incoming connections.
   @optparam timeout Optional wait time.
   @return A new TCPSocket after a succesful connection.
   @raise NetError on system error.

   This method accepts incoming connection and creates a TCPSocket object that
   can be used to communicate with the remote host. Before calling accept(), it is
   necessary to have succesfully called bind() to bind the listening application to
   a certain local address.

   If a timeout is not specified, the function will block until a TCP connection is
   received. If it is specificed, is a number of millisecond that will be waited
   before returning a nil. Setting the timeout to zero will cause accept to return
   immediately, providing a valid TCPSocket as return value only if an incoming
   connection was already pending.

   The wait blocks the VM, and thus, also the other coroutines.
   If a system error occours during the wait, a NetError is raised.
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
      vm->raiseModError( new  NetError( ErrorParam( FALSOCK_ERR_ACCEPT, __LINE__ ).
         desc( FAL_STR( sk_msg_erraccept ) ).sysError( (uint32) srvs->lastError() ) ) );
      return;
   }

   if ( skt == 0 ) {
      vm->retnil();
      return;
   }

   Item *tcp_class = vm->findWKI( "TCPSocket" );
   fassert( tcp_class != 0 );
   CoreObject *ret_s = tcp_class->asClass()->createInstance();
   ret_s->setUserData( skt );

   vm->retval( ret_s );
}


/*#
   @class NetError
   @brief Error generated by network related system failures.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   The error code can be one of the codes declared in the @a NetErrorCode enumeration.
   See the Error class in the core module.
*/

/*#
   @init NetError
   @brief Initializes the network error.
*/
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
