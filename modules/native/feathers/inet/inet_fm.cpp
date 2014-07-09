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

#define SRC "modules/native/feathres/inet/inet_fm.cpp"
#include "inet_mod.h"
#include "inet_fm.h"

#ifdef FALCON_SYSTEM_WIN
#include "winselectmpx.h"
#endif

#include <falcon/autocstring.h>
#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>



#include <errno.h>

/*#
   @beginmodule feathers.inet
*/

namespace Falcon {

static Mod::Address* internal_read_address( Function* func, Item* i_address )
{
   static Class* clsAddress = static_cast<Feathers::ModuleInet*>(func->fullModule())->addressClass();

   Mod::Address* target = 0;
   if( i_address != 0 )
   {
      if( i_address->isString() )
      {
         const String& sTarget = *i_address->asString();
         // this might throw
         target = new Mod::Address(sTarget);
      }
      else
      {
         // might return 0
         target = static_cast<Mod::Address*>(i_address->asParentInst( clsAddress ));
         target->incref();
      }
   }

   if( target == 0 )
   {
      throw func->paramError(__LINE__, SRC);
   }

   return target;
}

namespace Feathers {

namespace {

/*#
   @function getHostName
   @brief Retrieves the host name of the local machine.
   @return A string containing the local machine name.
   @raise NetError if the host name can't be determined.

   Returns the network name under which the machine is known to itself. By
   calling @a resolveAddress on this host name, it is possible to determine all
   the addresses of the interfaces that are available for networking.

   If the system cannot provide an host name for the local machine, a NetError
   is raised.

   @note If the function fails, it is possible to retrieve local addresses using
      through @a resolveAddress using the the name "localhost".
*/
FALCON_DECLARE_FUNCTION(getHostName, "")
FALCON_DEFINE_FUNCTION_P1(getHostName)
{
   String *s = new String;
   if ( ::Falcon::Mod::getHostName( *s ) )
   {
      ctx->returnFrame(FALCON_GC_HANDLE(s));
   }
   else {
      delete s;
      throw FALCON_SIGN_XERROR( NetError, FALSOCK_ERR_RESOLV,
               .desc(FALSOCK_ERR_RESOLV_MSG)
               .sysError( (uint32) errno ) );
   }
}

/*#
   @function haveSSL
   @brief Check if the socket module has SSL capabilities.
   @return True if we have SSL.
 */
FALCON_DECLARE_FUNCTION(haveSSL, "")
FALCON_DEFINE_FUNCTION_P1(haveSSL)
{
#if WITH_OPENSSL
   ctx->returnFrame( Item().setBoolean( true ) );
#else
   ctx->returnFrame( Item().setBoolean( false ) );
#endif
}

}

// ==============================================
// Class Socket
// ==============================================

namespace CSocket {
/*#
   @class Socket
   @brief Networking socket class.
   @optparam family Address family used by this socket.
   @optparam type Socket type.
   @optparam protocol Protocol used by the socket.
   @raise NetError if the socket creation fails.

   The Socket class implements a system level, network-oriented
   socket implementation.

   Family, type and protocol parameters can assume any value
   understood by the underlying system implementation of the
   socket interface.

   Some commonly used macros are provided by the host module,
   and have names identical to the macros declared by the POSIX
   standards. Check the module documentation for further details.

   When @b family is not specified or nil, it defaults to AF_INET6,
   if natively provided by the system. Otherwise, it defaults to
   AF_INET.

   When @b type is not specified or nil, it defaults to SOCK_STREAM.

   When @b protocol is not specified or nil, it defaults to 0; this means
   that the protocol is selected bsed on the address family/type
   combination.

   @note The default Socket() constructor is suitable to create server and
   client TCP sockets.

   @section Automatic IPv4 upgrade.

   On those systems where IPv6 is natively handled by the underlying system,
   if the socket address family is set to IPv6, IPv4 addresses are automatically
   translated by this class in IPv6 prior any address-sensible operation:
   - bind
   - connect
   - send (when specifying a target address).

   This means that @a Address instances resolved by the name resolution system
   as IPv4 entries don't need any form of manual conversion.
*/

/*#
@property address Socket
@brief Return the address associated with a socket.
*/
void get_address(const Class* cls, const String&, void* instance, Item& value )
{
   TRACE1( "get WOPI.pDataDir for %p", instance );
   ModuleInet* inet = static_cast<ModuleInet*>( cls->module() );
   Mod::Socket* socket = static_cast<Mod::Socket*>(instance);
   Mod::Address* a = socket->address();
   if( a  != 0 )
   {
      a->incref();
      value = FALCON_GC_STORE(inet->addressClass(), a);
   }
   else {
      value.setNil();
   }
}

/*#
@property type Socket
@brief Type of this socket
*/
void get_type(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.type for %p", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   value.setInteger( static_cast<int64>(sock->type()) );
}

/*#
@property closed Socket
@brief True if this socket was closed.
*/
void get_closed(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.closed for %p", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   value.setBoolean( sock->descriptor() == FALCON_INVALID_SOCKET_VALUE );
}

/*#
@property eof Socket
@brief True if this an end-of-stream signal is detected from remote.

This is set to true when a read operation returns 0, and this is not
due to a non-blocking operation returning zero data.
*/
void get_eof(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.closed for %p", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   value.setBoolean( sock->eof() );
}

/*#
@property stream Socket
@brief Access the stream interface for this socket.

If this socket is stream-oriented, this property holds a reference
to a Stream subclass instance. I/O operation on the stream map on stream-oriented
I/O operations on the socket, and the stream selectable interface maps to the
socket selectable status.
*/
void get_stream(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.Stream for %p", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);

   Stream* stream = sock->makeStreamInterface();
   value.setUser( stream->handler(), stream );
}


/*#
@property family Socket
@brief Address family of this socket
*/
void get_family(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.family for %p", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   value.setInteger( static_cast<int64>(sock->family()) );
}

/*#
@property protocol Socket
@brief Network protocol set at creation of this socket
*/
void get_protocol(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.protocol for %p", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   value.setInteger( static_cast<int64>(sock->protocol()) );
}

/*#
@property host Socket
@brief Shortcut returning the host string representation associated with a socket.

This will not invoke any resolution support. To have explicit reverse-address
resolution, get the @a Socket.address and then @a Address.resolve.
*/
void get_host(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.host for %p", instance );
   Mod::Socket* socket = static_cast<Mod::Socket*>(instance);
   Mod::Address* a = socket->address();

   if (a == 0)
   {
      value.setNil();
      return;
   }

   String* ret = new String;
   a->getHost(*ret);
   value = FALCON_GC_HANDLE(ret);
}

/*#
@property service Socket
@brief Service (port name) associated with the given socket.

This will not invoke any resolution support. To have explicit reverse-address
resolution, get the @a Socket.address and then @a Address.resolve.
*/
void get_service(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.service for %p", instance );
   Mod::Socket* socket = static_cast<Mod::Socket*>(instance);
   Mod::Address* a = socket->address();

   if (a == 0)
   {
      value.setNil();
      return;
   }

   String* ret = new String;
   a->getService(*ret);
   value = FALCON_GC_HANDLE(ret);
}

/*#
@property port Socket
@brief Numeric port associated with the given socket.

This will not invoke any resolution support. To have explicit reverse-address
resolution, get the @a Socket.address and then @a Address.resolve.
*/
void get_port(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.port for %p", instance );
   Mod::Socket* socket = static_cast<Mod::Socket*>(instance);
   Mod::Address* a = socket->address();

   if (a == 0)
   {
      value.setNil();
      return;
   }

   value = (int64) a->getPort();
}

/*#
@property descriptor Socket
@brief BSD/Posix descriptor representing this socket at O/S level
*/
void get_descriptor(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.descriptor for %p", instance );
   Mod::Socket* socket = static_cast<Mod::Socket*>(instance);
   value = (int64) socket->descriptor();
}

FALCON_DECLARE_FUNCTION(init, "family:[N],type:[N],protocol:[N]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   TRACE1( "ClassSocket::init(%d)", ctx->paramCount() );
   Item* i_family = ctx->param(0);
   Item* i_type = ctx->param(1);
   Item* i_protocol = ctx->param(2);

   Mod::Socket* sock = static_cast<Mod::Socket*>( ctx->self().asInst() );
   if( (i_family != 0 && !( i_family->isOrdinal() || i_family->isNil() ))
       ||(i_type != 0 && !( i_type->isOrdinal() || i_type->isNil() ))
       || (i_protocol != 0 && !( i_protocol->isOrdinal() || i_protocol->isNil() ))
   )
   {
      throw paramError(__LINE__, SRC );
   }

   /*
#ifdef AF_INET6
   int family = i_family == 0 || i_family->isNil() ? AF_INET6 : (int) i_family->forceInteger();
#else
   int family = i_family == 0 || i_family->isNil() ? AF_INET : (int) i_family->forceInteger();
#endif
   */

   int family = i_family == 0 || i_family->isNil() ? AF_INET : (int) i_family->forceInteger();

   int type = i_type == 0 || i_type->isNil() ? SOCK_STREAM : (int) i_type->forceInteger();
   int protocol = i_protocol == 0 || i_protocol->isNil() ? 0 : (int) i_protocol->forceInteger();

   sock->create( family, type, protocol );
   ctx->returnFrame(ctx->self());
}

/*#
   @method connect Socket
   @brief Connects this socket to a remote host.
   @param address Remote host to be connected to.
   @optparam Port or service name to be connected to.
   @raise NetError in case of underlying connection error.

   Connects with a remote listening TCP host. The operation may fail for a
   system error, in which case a NetError is raised.

   The connection attempt may timeout if it takes more than the time specified
   in @a Socket.setTimeout method. In that case, the @a Socket.isConnected method may check if the
   connection has been established at a later moment. So, it is possible to set the
   socket timeout to 0, and then check periodically for connection success without
   never blocking the VM.

   The host name may be a name to be resolved by the system resolver or it may
   be an already resolved dot-quad IP, or it may be an IPV6 address.

   @see Socket.setTimeout
*/
FALCON_DECLARE_FUNCTION(connect, "address:S|Address,async:[B]")
FALCON_DEFINE_FUNCTION_P1(connect)
{
   TRACE1( "Socket.Connect(%d params) for %p", ctx->paramCount(), ctx->self().asInst() );
   Mod::Address* target = internal_read_address(this, ctx->param(0) );
   Item* i_async = ctx->param(1);

   // resolve the address -- lightly if async, fully if sync
   bool bAsync = i_async != 0 ? i_async->isTrue() :  false;
   if( ! target->isResolved() && ! bAsync )
   {
      target->resolve(true);
   }
   // light resolution -- with error control -- is done automatically by Mod::Socket.

   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());
   try {
      sock->connect( target, bAsync );
      target->decref();
   }
   catch( ... )
   {
      target->decref();
      throw;
   }

   ctx->returnFrame();
}


/*#
   @method isConnected Socket
   @brief Check if this Socket is currently connected with a remote host.
   @return True if the socket is currently connected, false otherwise.
   @raise NetError if a the connection process caused an error in the meanwhile.

   This method checks if this Socket is currently connected with a remote host.

   @see Socket.connect
*/
FALCON_DECLARE_FUNCTION(isConnected, "")
FALCON_DEFINE_FUNCTION_P1(isConnected)
{
   Mod::Socket *tcps = static_cast<Mod::Socket*>( ctx->self().asInst() );
   ctx->returnFrame(Item().setBoolean(tcps->isConnected()));
}


static void internal_send( Function* caller, VMContext* ctx, Mod::Address* to )
{
   int base = to == 0 ? 0 : 1;
   Item *i_data = ctx->param( base + 0 );
   Item *i_start = ctx->param( base + 1 );
   Item *i_count = ctx->param( base + 2 );

   if ( i_data == 0 || ! i_data->isString()
        || ( i_count != 0 && ! i_count->isOrdinal() )
        || ( i_start != 0 && ! i_start->isOrdinal() )
      )
   {
      throw caller->paramError(__LINE__,SRC);
   }


   String* dataStr = i_data->asString();
   int64 len = (int64) dataStr->size();
   int64 start = i_start == 0 ? 0 : i_start->forceInteger();
   if( start < 0 )
   {
      start =  len + start;
   }

   // keep 0 valid, it's used to test writes.
   if( start > len )
   {
      ctx->returnFrame(Item().setInteger(0));
      return;
   }

   int64 count;
   if( i_count != 0 )
   {
       count = i_count->forceInteger();
       if( count + start > len )
       {
          count = len - start;
       }
   }
   else {
      count = len - start;
   }

   const byte* data = dataStr->getRawStorage();

   Mod::Socket *tcps = static_cast<Mod::Socket*>( ctx->self().asInst() );

   int64 res = (int64) tcps->send( data + start, (int) count, to );
   ctx->returnFrame(res);
}


/*#
   @method send Socket
   @brief Send data on the network connection.
   @param buffer The buffer containing the data to be sent.
   @optparam start Begin position in the buffer (in bytes).
   @optparam count Amount of bytes to be sent.
   @return Number of bytes actually sent through the network layer.
   @raise NetError on network error,

   The @b buffer may be a byte-only string or a
   byte-wide MemBuf; it is possible to send also multibyte strings (i.e. strings
   containing international characters) or multi-byte memory buffers, but in that
   case the sent data may get corrupted as a transmission may deliver only part
   of a character or of a number stored in a memory buffer.

   When using a @b MemBuf item type, the function will try to send the data
   between @a MemBuf.position and @a MemBuf.limit. After a correct write,
   the position is moved forward accordingly to the amount of bytes sent.

   If a @b size parameter is not specified, the method will try to send the whole
   content of the buffer, otherwise it will send at maximum size bytes.

   If a @b start parameter is specified, then the data sent will be taken starting
   from that position in the buffer (counting in bytes from the start). This is
   useful when sending big buffers in several steps, so that
   it is not necessary to create substrings for each send, sparing both
   CPU and memory.

   @note The @b start and @b count parameters are ignored when using a memory
   buffer.

   The returned value may be 0 in case of timeout, otherwise it will be a
   number between 1 and the requested size. Programs should never assume
   that a successful @b send has sent all the data.

   In case of error, a @a NetError is raised.

   @see Socket.setTimeout
*/
FALCON_DECLARE_FUNCTION(send, "buffer:S,start:[N],count:[N]")
FALCON_DEFINE_FUNCTION_P1(send)
{
   internal_send( this, ctx, 0 );
}


/*#
   @method sendTo Socket
   @brief Sends a datagram to a given address.
   @param host Remote host where to send the datagram.
   @param service Remote service or port number where to send the datagram.
   @param buffer The buffer to be sent.
   @optparam size Amount of bytes from the buffer to be sent.
   @optparam start Begin position in the buffer.
   @raise NetError on network error.

   This method works as the Socket.send method, with the
   main difference that the outgoing datagram can be directed towards a
   specified @b host, and that a whole datagram is always completely
   filled before being sent, provided that the specified size
   does not exceed datagram size limits.

   The @b host parameter may be an host name to be resolved or an address;
   if the @a UDPSocket.broadcast method has been successfully called,
   it may be also a multicast or broadcast address.

   The @b service parameter is a string containing either a service name
   (i.e. "http") or  a numeric port number (i.e. "80", as a string).

   The @b buffer may be a byte-only string or a
   byte-wide MemBuf; it is possible to send also multibyte strings (i.e. strings
   containing international characters) or multi-byte memory buffers, but in that
   case the sent data may get corrupted as a transmission may deliver only part
   of a character or of a number stored in a memory buffer.

   @note If the @b buffer is a MemBuf item, @b size and @b start parameters are
   ignored, and the buffer @b MemBuf.position and @b MemBuf.limit are used
   to determine how much data can be received. After a successful receive,
   the value of @b MemBuf.position is moved forward accordingly.

   If a @b size parameter is not specified, the method will try to send the whole
   content of the buffer, otherwise it will send at maximum size bytes. If a
   @b start parameter is specified, then the data sent will be taken starting
   from that position in the buffer (counting in bytes from the start).

   This is useful when sending big buffers in several steps, so that
   it is not necessary to create substrings for each send, sparing both
   CPU and memory.

   The returned value may be 0 in case of timeout, otherwise it will be a
   number between 1 and the requested size. Programs should never assume
   that a successful @b sendTo has sent all the data.

   In case of error, a @a NetError is raised.

   @see Socket.setTimeout
*/
FALCON_DECLARE_FUNCTION(sendTo, "address:S|Address,buffer:S,start:[N],count:[N]")
FALCON_DEFINE_FUNCTION_P1(sendTo)
{
   Mod::Address* address = internal_read_address(this, ctx->param(0));

   try {
      internal_send( this, ctx, address );
      address->decref();
   }
   catch( ... )
   {
      address->decref();
      throw;
   }
}



/*#
   @method recv Socket
   @brief Reads incoming data.
   @param buffer A (possibly pre-allocated)  buffer to fill.
   @optparam size Maximum size in bytes to be read.
   @return Amount of bytes actually read.
   @raise NetError on network error.

   The @b buffer parameter is a buffer to be filled: a @b MemBuf or a an empty
   string (for example, a string created with @a strBuffer).

   If the @b size parameter is provided, it is used to define how many bytes can
   be read and stored in the buffer.

   If the @b buffer parameter is a string, it is automatically resized to fit
   the incoming data. On a successful read, it's size will be trimmed to the
   amount of read data, but the internal buffer will be retained; successive reads
   will reuse the already available data buffer. For example:

   @code
   str = ""
   while mySocket.recv( str, 1024 ) > 0
      > "Read: ", str.len()
      ... do something with str ...
   end
   @endcode

   This allocates 1024 bytes in str, which is trimmed each time to the amount
   of data really received, but is never re-allocated during this loop. However, this
   is more efficient as you spare a parameter in each call, but it makes less
   evident how much data you want to receive:

   @code
   str = strBuffer( 1024 )

   while mySocket.recv( str ) > 0
      > "Read: ", str.len()
      ... do something with str ...
   end
   @endcode

   If the @b buffer parameter is a MemBuf, the read data will be placed at
   @a MemBuf.position. After a successful read, up to @a MemBuf.limit bytes are
   filled, and @a MemBuf.position is advanced. To start processing the data in the
   buffer, use @a MamBuf.flip().

   In case of system error, a NetError is raised.

   @see Socket.setTimeout
*/
FALCON_DECLARE_FUNCTION(recv, "buffer:S,count:[N]")
FALCON_DEFINE_FUNCTION_P1(recv)
{
   Item *i_recv = ctx->param(0);
   Item *i_size = ctx->param(1);
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());

   if( i_recv == 0 || ! i_recv->isString()
       || ( i_size != 0 && ! i_size->isOrdinal() ) )
   {
      throw paramError(__LINE__, SRC);
   }

   String* buffer = i_recv->asString();
   if( buffer->isImmutable() )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_param_type, .extra( "Immutable string") );
   }
   length_t size;
   buffer->clear();
   if (i_size == NULL)
   {
      size = buffer->size();
   }
   else
   {
      size = (length_t) i_size->forceInteger();
      buffer->reserve( size );
   }
   buffer->toMemBuf();

   int64 res = (int64) sock->recv( buffer->getRawStorage(), size, 0 );
   buffer->size((length_t)res);
   ctx->returnFrame( res );
}


FALCON_DECLARE_FUNCTION(recvFrom, "address:Address,buffer:S,count:[N]")
FALCON_DEFINE_FUNCTION_P1(recvFrom)
{
   static Class* clsAddress = static_cast<ModuleInet*>(fullModule())->addressClass();

   Item *i_addr = ctx->param(0);
   Item *i_recv = ctx->param(1);
   Item *i_size = ctx->param(2);
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());

   Mod::Address* addr;
   if( i_addr == 0 || (addr = static_cast<Mod::Address*>(i_addr->asParentInst( clsAddress ) )) == 0
       || i_recv == 0 || ! i_recv->isString()
       || ( i_size != 0 && ! i_size->isOrdinal() ) )
   {
      throw paramError(__LINE__, SRC);
   }

   String* buffer = i_recv->asString();
   if( buffer->isImmutable() )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_param_type, .extra( "Immutable string") );
   }

   buffer->clear();
   length_t size = (length_t) i_size->forceInteger();
   buffer->reserve( size );
   buffer->toMemBuf();

   int64 res = (int64) sock->recv( buffer->getRawStorage(), size, addr );
   buffer->size((length_t)res);
   ctx->returnFrame( res );
}

/*#
   @method closeRead Socket
   @brief Closes a socket read side.
   @raise NetError in case of underlying connection error during the closing phase.

   Closes the socket read side, discarding incoming messages and notifying
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
FALCON_DECLARE_FUNCTION(closeRead, "")
FALCON_DEFINE_FUNCTION_P1(closeRead)
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());
   sock->closeRead();
   ctx->returnFrame();
}


/*#
   @method closeWrite Socket
   @brief Closes a socket write side.
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
FALCON_DECLARE_FUNCTION(closeWrite, "")
FALCON_DEFINE_FUNCTION_P1(closeWrite)
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());
   sock->closeWrite();
   ctx->returnFrame();
}



/*#
   @method close Socket
   @brief Closes the socket.
   @raise NetError in case of underlying connection error during the closing phase.

   Closes the socket, discarding incoming messages and notifying the remote side
   about the event. The close message must be acknowledged by the remote host, so
   the function may actually fail, block and/or timeout - see setTimeout() .

   In case of error, a NetError is raised, while in case of timeout @b false is returned.
   On successful completion @b true is returned.

   @see Socket.setTimeout
*/
FALCON_DECLARE_FUNCTION(close, "")
FALCON_DEFINE_FUNCTION_P1(close)
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());
   sock->close();
   ctx->returnFrame();
}


/*#
   @method setOpt Socket
   @brief Sets a socket option.
   @param level The option level.
   @param option The option.
   @param value The new value.
   @raise NetError in case error setting the option.

   This method configures the given option at the given protocol
   level for this socket.

   Some well-known options are supported, and the type of the
   option is checked prior being set. If an unknown level/option
   combination is set, an integer value only will be allowed.

   The list of the known levels and options is given in the
   module description.

   @note The SOL_SOCKET/SO_LINGER option is treated specially.
   It takes an integer value; if its less than zero, the linger
   option is disabled. If it's zero or greater, the linger option
   is enabled and the given value is set as linger value.

*/
FALCON_DECLARE_FUNCTION(setOpt, "level:N,option:N,value:X")
FALCON_DEFINE_FUNCTION_P1(setOpt)
{
   Item* i_level = ctx->param(0);
   Item* i_option = ctx->param(1);
   Item* i_value = ctx->param(2);

   if( i_level == 0 || ! i_level->isOrdinal()
      || i_option == 0 || ! i_option->isOrdinal()
      || i_value == 0
   )
   {
      throw paramError(__LINE__, SRC );
   }

   int32 level = (int32) i_level->forceInteger();
   int32 option = (int32) i_option->forceInteger();
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());

   sock->setOption( level, option, *i_value );
   ctx->returnFrame();
}

/*#
   @method getOpt Socket
   @brief Gets a socket option.
   @param level The option level.
   @param option The option.
   @return The current value of the option.
   @raise NetError in case error setting the option.

   This method reads the current value of the given option at the given protocol
   level for this socket.

   Some well-known options are supported, and the type of the
   option is specially mangled prior being returned.
   If an unknown level/option
   combination is get, the resulting value will always be an integer.

   The list of the known levels and options is given in the
   module description.

   @note The SOL_SOCKET/SO_LINGER option is treated specially.
   if the linger is disabled, the method returns -1; if it's
   enabled, it returns the linger size (which might be 0).
*/
FALCON_DECLARE_FUNCTION(getOpt, "level:N,option:N")
FALCON_DEFINE_FUNCTION_P1(getOpt)
{
   Item* i_level = ctx->param(0);
   Item* i_option = ctx->param(1);

   if( i_level == 0 || ! i_level->isOrdinal()
      || i_option == 0 || ! i_option->isOrdinal()
   )
   {
      throw paramError(__LINE__, SRC );
   }

   int32 level = (int32) i_level->forceInteger();
   int32 option = (int32) i_option->forceInteger();
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());

   Item result;
   sock->getOption( level, option, result );
   ctx->returnFrame(result);
}


/*#
   @method bind Socket
   @brief Binds this socket to the given address.
   @param addr A string representing an address or an @a Address class instance.
   @raise NetError If the socket cannot be bound to the given address.

*/
FALCON_DECLARE_FUNCTION(bind, "addr:S|Address")
FALCON_DEFINE_FUNCTION_P1(bind)
{
   Mod::Address* target = internal_read_address(this, ctx->param(0));
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());

   try {
      sock->bind(target);
      target->decref();
   }
   catch(...)
   {
      target->decref();
      throw;
   }
}

/*#
   @method listen Socket
   @brief Declares a socket ready to accept connections.
   @optparam size Number of pending connection allowed before refusing a new one.
   @raise NetError If the socket cannot be put in listen status.

*/
FALCON_DECLARE_FUNCTION(listen, "size:[N]")
FALCON_DEFINE_FUNCTION_P1(listen)
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());
   Item* i_size = ctx->param(0);
   if( i_size != 0 && ! i_size->isOrdinal() )
   {
      throw paramError( __LINE__, SRC );
   }

   int size = (int) i_size->forceInteger();
   sock->listen( size );
}

/*#
   @method accept Socket
   @brief Dequeues an incoming client that is being connected.
   @return a new Socket that has been accepted, or 0 if no socket could be dequeued.
   @raise NetError If the socket is not in listen status, or if there are other network error.

   This method will wait till a new client files a connection to this socket; if this
   has already happened, the method will return immediately the new client.

   A socket in listen state might be put in a selector to detect when it has new clients
   incoming. The socket will be signaled ready for write when this happens.

   If this socket is non-blocking, or if it was waken up during a spurious wakeup in a
   selector, the method might return nil; this is not to be considered an error. When
   this happens, the owner should check if the socket has been closed asynchronously
   by accessing the closed property.
*/
FALCON_DECLARE_FUNCTION(accept, "")
FALCON_DEFINE_FUNCTION_P1(accept)
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());

   Mod::Socket* newSkt = sock->accept();
   if( newSkt == 0 )
   {
      ctx->returnFrame();
   }
   else
   {
      ModuleInet* mod = static_cast<ModuleInet*>(fullModule());
      ctx->returnFrame( FALCON_GC_STORE(mod->socketClass(), newSkt ) );
   }
}


/*#
   @property broadcasting Socket
   @brief Activates broadcasting and multicasting abilities on this UDP socket.
   @raise NetError on system error.

   @note This is normally set to false.
   The socket must have already be bound as datagram socket with @a Socket.bind
   to activate broadcasting.
*/
void get_broadcasting(const Class*, const String&, void* instance, Item& value )
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   value.setBoolean( sock->broadcasting() );
}


void set_broadcasting(const Class*, const String&, void* instance, const Item& value )
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   sock->broadcasting(value.isTrue());
}

/*#
   @property nonblocking Socket
   @brief Check or set nonblocking mode on this socket.
   @raise NetError on system error.
*/
void get_nonblocking(const Class*, const String&, void* instance, Item& value )
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   value.setBoolean( sock->isNonBlocking() );
}


void set_nonblocking(const Class*, const String&, void* instance, const Item& value )
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   sock->setNonBlocking(value.isTrue());
}


#if WITH_OPENSSL
/*#
   @method sslConfig Socket
   @brief Prepare a socket for SSL operations.
   @param serverSide True for a server-side socket, false for a client-side socket.
   @param version SSL method (one of SSLv2, SSLv3, SSLv23, TLSv1, DTLSv1 ). Default SSLv3
   @optparam cert Certificate file
   @optparam pkey Private key file
   @optparam ca Certificate authorities file

   Must be called after socket is really created, that is after connect() is called.
 */
FALCON_DECLARE_FUNCTION(sslConfig, "serverSide:B,version:S,cert:[S],pkey:[S],ca:[S]")
FALCON_DEFINE_FUNCTION_P1(sslConfig)
{
   Mod::Socket *tcps = static_cast<Mod::Socket *>( ctx->self().asInst() );

   Item* i_asServer = ctx->param( 0 );
   Item* i_sslVer = ctx->param( 1 );
   Item* i_cert = ctx->param( 2 );
   Item* i_pkey = ctx->param( 3 );
   Item* i_ca = ctx->param( 4 );

   if ( !i_asServer || !( i_asServer->isBoolean() )
      || !i_sslVer || !( i_sslVer->isInteger() )
      || ( i_cert && !( i_cert->isString() || i_cert->isNil() ) )
      || ( i_pkey && !( i_pkey->isString() || i_pkey->isNil() ) )
      || ( i_ca && !( i_ca->isString() || i_ca->isNil() ) ) )
   {
      throw  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "B,I,[S,S,S]" ) );
   }

   String* s_cert = i_cert && !i_cert->isNil() ? i_cert->asString() : 0;
   String* s_pkey = i_pkey && !i_pkey->isNil() ? i_pkey->asString() : 0;
   String* s_ca = i_ca && !i_ca->isNil() ? i_ca->asString() : 0;

   tcps->sslConfig( i_asServer->asBoolean(),
                        (Mod::SSLData::sslVersion_t) i_sslVer->asInteger(),
                        s_cert, s_pkey, s_ca );
   ctx->returnFrame();
}


/*#
   @method sslConnect Socket
   @brief Negotiate an SSL connection.

   Must be called after socket is connected and has been properly configured for
   SSL operations.
 */
FALCON_DECLARE_FUNCTION(sslConnect, "")
FALCON_DEFINE_FUNCTION_P1(sslConnect)
{
   Mod::Socket *tcps = static_cast<Mod::Socket *>( ctx->self().asInst() );
   tcps->sslConnect();
   ctx->returnFrame();
}


/*#
   @method sslClear Socket
   @brief Free resources taken by SSL contexts.

   Useful if you want to reuse a socket.
 */
FALCON_DECLARE_FUNCTION(sslClear, "")
FALCON_DEFINE_FUNCTION_P1(sslClear)
{
   Mod::Socket *tcps = static_cast<Mod::Socket *>( ctx->self().asInst() );
   tcps->sslClear();
   ctx->returnFrame();
}
#endif // WITH_OPENSSL

} /* NS CSocket */

ClassSocket::ClassSocket():
     Class("Socket")
{
   this->m_bHasSharedInstances = true;

   setConstuctor( new CSocket::Function_init );

   addProperty("address",&CSocket::get_address);
   addProperty("descriptor",&CSocket::get_descriptor);
   addProperty("host",&CSocket::get_host);
   addProperty("service",&CSocket::get_service);
   addProperty("port",&CSocket::get_port);
   addProperty("closed",&CSocket::get_closed);
   addProperty("eof",&CSocket::get_eof);
   addProperty("stream",&CSocket::get_stream, 0, false, true );

   addProperty("broadcasting", &CSocket::get_broadcasting, &CSocket::set_broadcasting );
   addProperty("nonblocking", &CSocket::get_nonblocking, &CSocket::set_nonblocking );

   addProperty("family",&CSocket::get_family);
   addProperty("type",&CSocket::get_type);
   addProperty("protocol",&CSocket::get_protocol);

   addMethod( new CSocket::Function_connect );
   addMethod( new CSocket::Function_isConnected );

   addMethod( new CSocket::Function_bind );
   addMethod( new CSocket::Function_listen );
   addMethod( new CSocket::Function_accept );

   addMethod( new CSocket::Function_recv );
   addMethod( new CSocket::Function_recvFrom );
   addMethod( new CSocket::Function_send );
   addMethod( new CSocket::Function_sendTo );
   addMethod( new CSocket::Function_closeRead );
   addMethod( new CSocket::Function_closeWrite );
   addMethod( new CSocket::Function_close );

   addMethod( new CSocket::Function_getOpt );
   addMethod( new CSocket::Function_setOpt );

#if WITH_OPENSSL
   addMethod( new CSocket::Function_sslConfig );
   addMethod( new CSocket::Function_sslConnect );
   addMethod( new CSocket::Function_sslClear );
#endif
}

ClassSocket::~ClassSocket()
{}

int64 ClassSocket::occupiedMemory( void* ) const
{
   // account for the socket structure in O/S (roughly)
   return sizeof(Mod::Socket)+32;
}


void ClassSocket::dispose( void* instance ) const
{
   TRACE1( "ClassSocket::dispose(%p)", instance );

   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   sock->decref();
}

void* ClassSocket::clone( void* instance ) const
{
   TRACE1( "ClassSocket::clone(%p)", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   sock->incref();
   return sock;
}

void* ClassSocket::createInstance() const
{
   return new Mod::Socket;
}

Selectable* ClassSocket::getSelectableInterface( void* instance ) const
{
   TRACE1( "ClassSocket::getSelectableInterface(%p)", instance );
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   return new Mod::SocketSelectable(this, sock);
}




/*#
   @method accept TCPServer
   @brief Waits for incoming connections.
   @optparam timeout Optional wait time.
   @return A new Socket after a successful connection.
   @raise NetError on system error.

   This method accepts incoming connection and creates a Socket object that
   can be used to communicate with the remote host. Before calling accept(), it is
   necessary to have successfully called bind() to bind the listening application to
   a certain local address.

   If a timeout is not specified, the function will block until a TCP connection is
   received. If it is specified, is a number of millisecond that will be waited
   before returning a nil. Setting the timeout to zero will cause accept to return
   immediately, providing a valid Socket as return value only if an incoming
   connection was already pending.

   The wait blocks the VM, and thus, also the other coroutines.
   If a system error occurs during the wait, a NetError is raised.
*/

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

//=======================================================================
// Class Address
//=======================================================================

namespace CAddress {

/*#
@property host Address
@brief Shortcut returning the host.
*/
void get_host(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Address.host for %p", instance );
   Mod::Address* addr = static_cast<Mod::Address*>(instance);

   String* ret = new String;
   addr->getHost(*ret);
   value = FALCON_GC_HANDLE(ret);
}

/*#
@property service Socket
@brief Service (port name) associated with the given socket.

This will not invoke any resolution support. To have explicit reverse-address
resolution, get the @a Socket.address and then @a Address.resolve.
*/
void get_service(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Address.service for %p", instance );
   Mod::Address* addr = static_cast<Mod::Address*>(instance);

   String* ret = new String;
   addr->getService(*ret);
   value = FALCON_GC_HANDLE(ret);
}

/*#
@property port Socket
@brief Numeric port associated with the given socket.

This will not invoke any resolution support. To have explicit reverse-address
resolution, get the @a Socket.address and then @a Address.resolve.
*/
void get_port(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Address.port for %p", instance );
   Mod::Address* addr = static_cast<Mod::Address*>(instance);
   value = (int64) addr->getPort();
}

/*#
@property resolved Socket
@brief True if the address has been resolved.

*/
void get_resolved(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Address.resolved for %p", instance );
   Mod::Address* addr = static_cast<Mod::Address*>(instance);
   value.setBoolean(addr->isResolved());
}

/*#
@property resolved Socket
@brief An array containing the resolved entities.

Will be nil if the address has not been resolved.
*/
void get_addresses(const Class*, const String&, void* instance, Item& value )
{
   TRACE1( "Address.resolved for %p", instance );
   Mod::Address* addr = static_cast<Mod::Address*>(instance);
   if( addr->isResolved() )
   {
      ItemArray* array = new ItemArray;
      int32 count = addr->getResolvedCount();
      for( int32 i = 0; i < count; ++i )
      {
         String *res = new String;
         String host, service;
         int port = 0;
         int family = 0;
         addr->getResolvedEntry( i, host, service, port, family );
         if( host.find(':') != String::npos )
         {
            *res = "[" + host + "]:";
            if( port != 0 )
            {
               res->N(port);
            }
            else if( ! service.empty() && service != "0")
            {
               *res += service;
            }
         }
         else
         {
            if( port != 0 )
            {
               *res = host + ":";
               res->N(port);
            }
            else if( ! service.empty() && service != "0")
            {
               *res = host + ":" + service;
            }

            else {
               *res = host;
            }
         }

         array->append(FALCON_GC_HANDLE(res));
      }
      value = FALCON_GC_HANDLE( array );
   }
   else {
      value.setNil();
   }
}

/*#
   @method set Address
   @brief Changes the network address stored in this Address entity.
   @param addr A full IPv4 or IPv6 address, or a host portion if @b srv has a value
   @optparam srv Service (text name) or IP port (numeric).
   @raise NetError If the given address is not valid.
*/
FALCON_DECLARE_FUNCTION(set, "addr:S,srv:[S|N]")
FALCON_DEFINE_FUNCTION_P1(set)
{
   Item* i_addr = ctx->param(0);
   Item* i_srv = ctx->param(1);

   if( i_addr == 0 || ! i_addr->isString()
       || (i_srv != 0 && !( i_srv->isString() || i_srv->isOrdinal()) )
       )
   {
      throw paramError(__LINE__, SRC);
   }

   Mod::Address* addr = static_cast<Mod::Address*>(ctx->self().asInst());
   if( i_srv == 0 )
   {
      addr->parse(*i_addr->asString());
   }
   else {
      if( i_srv->isString() )
      {
         addr->set(*i_addr->asString(), *i_srv->asString() );
      }
      else if( i_srv->isInteger() )
      {
         String temp;
         temp.N(i_srv->forceInteger());
         addr->set(*i_addr->asString(), temp);
      }
   }

   ctx->returnFrame();
}

/*#
   @method resolve Address
   @brief Tries to perform name resolution on this address.
   @optparam async True to make the request asynchronous.
   @raise NetError If the resolution fails.

   Normally, the request is synchronous.

   In asynchronous mode, the method returns
   immediately, and this Address becomes signaled (waitable) as
   the resolution is performed.

   Set @b sync to false to have this method to wait for the
   results prior returning the control to the caller.
*/
FALCON_DECLARE_FUNCTION(resolve, "sync:[B]")
FALCON_DEFINE_FUNCTION_P1(resolve)
{
   Item* i_sync = ctx->param(0);
   Mod::Address* addr = static_cast<Mod::Address*>(ctx->self().asInst());
   ModuleInet* mod = static_cast<ModuleInet*>(fullModule());
   bool async = i_sync == 0 ? false: i_sync->isTrue();
   if( async )
   {
      ContextManager* mgr = &ctx->vm()->contextManager();
      Mod::Resolver* res = new Mod::Resolver( mgr, mod->resolverClass() );
      res->startResolving( addr );

      ctx->returnFrame( FALCON_GC_STORE(mod->resolverClass(), res) );
   }
   else {
      addr->resolve();
      ctx->returnFrame();
   }
}


} /* end of NS CAddress */

ClassAddress::ClassAddress():
         Class("Address")
{
   this->m_bHasSharedInstances = true;

   addProperty( "host", &CAddress::get_host );
   addProperty( "port", &CAddress::get_port );
   addProperty( "service", &CAddress::get_service );
   addProperty( "resolved", &CAddress::get_resolved );
   addProperty( "addresses", &CAddress::get_addresses );

   addMethod( new CAddress::Function_set );
   addMethod( new CAddress::Function_resolve );
}

ClassAddress::~ClassAddress()
{}

int64 ClassAddress::occupiedMemory( void* instance ) const
{
   Mod::Address* address = static_cast<Mod::Address*>(instance);
   return address->occupiedMemory();
}

void ClassAddress::dispose( void* instance ) const
{
   Mod::Address* address = static_cast<Mod::Address*>(instance);
   address->decref();
}

void* ClassAddress::clone( void* instance ) const
{
   Mod::Address* address = static_cast<Mod::Address*>(instance);
   Mod::Address* copy = new Mod::Address(*address);
   return copy;
}

void* ClassAddress::createInstance() const
{
   return new Mod::Address;
}

void ClassAddress::describe( void* instance, String& target, int, int ) const
{
   Mod::Address* address = static_cast<Mod::Address*>(instance);
   address->toString(target);
}


bool ClassAddress::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   Mod::Address* address = static_cast<Mod::Address*>(instance);
   if( pcount >= 2 )
   {
      Item& i_host = ctx->opcodeParam(1);
      Item& i_port = ctx->opcodeParam(0);
      if( ! i_host.isString() || ! (i_port.isOrdinal() || i_port.isString()) )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra("S,[N|S]"));
      }

      if (i_port.isOrdinal())
      {
         String temp;
         temp.N(i_port.forceInteger());
         address->set( *i_host.asString(), temp );
      }
      else {
         address->set( *i_host.asString(), *i_port.asString() );
      }
   }
   else if( pcount == 1 )
   {
      Item& i_host = ctx->opcodeParam(0);
      if( ! i_host.isString() )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra("S,[N|S]"));
      }
      address->parse( *i_host.asString() );
   }
   else {
      throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra("S,[N|S]"));
   }

   return false;
}


//=================================================================
// Resolver
//=================================================================

static void get_address(const Class* cls, const String&, void* instance, Item& value )
{
   Mod::Resolver* res = static_cast<Mod::Resolver*>(instance);
   ModuleInet* mod = static_cast<ModuleInet*>(cls->module());
   res->address()->incref();
   value = FALCON_GC_STORE( mod->addressClass(), res->address() );
}

ClassResolver::ClassResolver():
         ClassShared("Resolver")
{
   setParent(Engine::instance()->stdHandlers()->sharedClass());
   addProperty("address", &get_address );
}

ClassResolver::~ClassResolver()
{
}

int64 ClassResolver::occupiedMemory( void* ) const
{
   return sizeof(Mod::Resolver);
}

void ClassResolver::dispose( void* instance ) const
{
   Mod::Resolver* res = static_cast<Mod::Resolver*>(instance);
   res->decref();
}

void* ClassResolver::clone( void* instance ) const
{
   Mod::Resolver* res = static_cast<Mod::Resolver*>(instance);
   res->incref();
   return res;
}

void* ClassResolver::createInstance() const
{
   return 0;
}

void ClassResolver::describe( void* instance, String& target, int , int  ) const
{
   Mod::Resolver* res = static_cast<Mod::Resolver*>(instance);
   String str;
   res->address()->toString(str);
   target = "Resolver{" + str + "}";
}

static void get_socket(const Class* cls, const String&, void* instance, Item& value )
{
   Mod::SocketStream* res = static_cast<Mod::SocketStream*>(instance);
   ModuleInet* mod = static_cast<ModuleInet*>(cls->module());
   // of course, the socket already exists in the GC, as it created us.
   value.setUser(mod->addressClass(), res->skt() );
}

ClassSocketStream::ClassSocketStream():
         ClassStream("SocketStream")
{
   setParent( Engine::instance()->stdHandlers()->streamClass() );
   addProperty("socket", &get_socket );
}

ClassSocketStream::~ClassSocketStream()
{
}

int64 ClassSocketStream::occupiedMemory( void* ) const
{
   return sizeof(Mod::SocketStream) + 16;
}

void* ClassSocketStream::clone( void* instance ) const
{
   Mod::SocketStream* stream = static_cast<Mod::SocketStream*>(instance);
   return stream->clone();
}


Selectable* ClassSocketStream::getSelectableInterface( void* instance ) const
{
   Mod::SocketStream* stream = static_cast<Mod::SocketStream*>(instance);
   return new Mod::SocketStream::Selectable( stream );
}



//=================================================================
// Module
//=================================================================

ModuleInet::ModuleInet():
         Module(FALCON_FEATHER_INET_NAME)
{
   m_clsAddress = new ClassAddress;
   m_clsSocket = new ClassSocket;
   m_clsResolver = new ClassResolver;
   m_clsSocketStream = new ClassSocketStream;

   #ifdef FALCON_SYSTEM_WIN
   m_smpxf = new Mod::WinSelectMPXFactory;
   #endif

   Mod::SocketStream::setHandler( m_clsSocketStream );

   *this
      << m_clsSocketStream
      << m_clsAddress
      << m_clsSocket
      << m_clsResolver
      << new ClassNetError
      ;

   //====================================
   // Net error code enumeration

   /*#
      @module inet
      @brief Advanced wrapping of the BSD Socket networking library.

      This module wraps the BSD Socket networking library, and adds
      additional support to help the integration of networking code
      into Falcon parallel programming.

      It provides mainly three classes:
      - Socket: wrapping a BSD socket entity.
      - Address: Representing a concrete instance of a network address,
                 and providing support for address resolution.
      - NetError: Error raised by various network-related functions.

      @section inet_error_codes Error codes

      The following error codes are returned as @a Error.code in NetError
      raised by this module.

      - @b ERR_GENERIC: A generic failure prevented the network layer to work
                     altogether. I.e. it was not possible to initialize
                     the network layer
      - @b ERR_RESOLV: An error happened while trying to resolve a network
                  address; possibly, the name resolution service was
                  not available or failed altogether.
      - @b ERR_CREATE: It was impossible to create the socket.
      - @b ERR_SEND: The network had an error while trying to send data.
      - @b ERR_CONNECT: The network had an error while receiving data from a remote host.
      - @b ERR_CLOSE: An error was detected while closing the socket. Either the socket
                  could not be closed (i.e. because it was already invalid) or the
                  close sequence was disrupted by a network failure.
      - @b ERR_BIND: The required address could not be allocated by the calling process.
                 Either the address is already busy or the bind operation required
                 privileges not owned by the process.
      - @b ERR_ACCEPT: The network system failed while accepting an incoming connection.
                 This usually means that the accepting thread has become unavailable.

      - @b ERR_ADDRESS:
      - @b ERR_FCNTL:
      - @b ERR_INCOMPATIBLE:
      - @b ERR_LISTEN:
      - @b ERR_RECV:
      - @b ERR_UNRESOLVED:
      - @b ERR_SSLCONFIG:
      - @b ERR_SSLCONNECT:
      - @b ERR_ALREADY_CREATED:

    @section inet_af_macros Address Family constants

       The following constants can be used as address family for socket creation,
       and are returned when querying the socket type.

       - AF_INET
       - AF_INET6
       - AF_IPX
       - AF_UNIX
       - AF_LOCAL

       Although this module provides this macros, any integer value can be used
       as long as this is known by the underlying system socket() function.

    @section inet_sock_macros Socket type constants.

        The following macros can be used as socket type for socket creation;

       - SOCK_DGRAM
       - SOCK_STREAM
       - SOCK_RAW
       - SOCK_SEQPACKET (on system where this feature is available).

       Moreover, a SOCK_NONE macro is provided, and returned by @a Socket.type
       for closed or non correctly configured sockets.

       Although this module provides this macros, any integer value can be used
       as long as this is known by the underlying system socket() function.

    @section inet_opts_macros Socket option constants
       The module defines the following options for the "option level" parameter
       in @a Socket.getOpt and @a Socket.setOpt.

      - SOL_SOCKET
      - IPPROTO_IP
      - IPPROTO_IPV6
      - IPPROTO_RAW
      - IPPROTO_TCP
      - IPPROTO_UDP

      The following constants are defined and can be used as "option value"
      parameter in @a Socket.getOpt and @a Socket.setOpt, when SOL_SOCKET is
      used as level. Depending on the
      option/level type, different values types are returned by @a Socket.getOpt,
      and different types are accepted as option value in @a Socket.setOpt.

       - SO_DEBUG: boolean
       - SO_REUSEADDR: boolean
       - SO_TYPE: integer
       - SO_ERROR: integer
       - SO_DONTROUTE: boolean
       - SO_BROADCAST: boolean
       - SO_KEEPALIVE: boolean
       - SO_RCVBUF: integer
       - SO_OOBINLINE: integer
       - SO_LINGER: integer -- see remarks on linger options in Socket.setOpt
       - SO_RCVLOWAT: integer
       - SO_SNDLOWAT: integer
       - SO_RCVTIMEO: integer
       - SO_SNDTIMEO: integer

      However, any combination of level/option is accepted, and passed to the
      underlying BSD socket setSockOption/getSockOption implementation.

      When an unknown option is set, numeric parameters only are accepted. When
      an option is read, the value is returned as a numeric parameter.
   */
   this->addConstant( "ERR_GENERIC", FALSOCK_ERR_GENERIC );
   this->addConstant( "ERR_ACCEPT", FALSOCK_ERR_ACCEPT );
   this->addConstant( "ERR_ADDRESS", FALSOCK_ERR_ADDRESS );
   this->addConstant( "ERR_BIND", FALSOCK_ERR_BIND );
   this->addConstant( "ERR_CLOSE", FALSOCK_ERR_CLOSE );
   this->addConstant( "ERR_CONNECT", FALSOCK_ERR_CONNECT );
   this->addConstant( "ERR_CREATE", FALSOCK_ERR_CREATE );
   this->addConstant( "ERR_FCNTL", FALSOCK_ERR_FCNTL );
   this->addConstant( "ERR_INCOMPATIBLE", FALSOCK_ERR_INCOMPATIBLE );
   this->addConstant( "ERR_LISTEN", FALSOCK_ERR_LISTEN );
   this->addConstant( "ERR_RECV", FALSOCK_ERR_RECV );
   this->addConstant( "ERR_RESOLV", FALSOCK_ERR_RESOLV );
   this->addConstant( "ERR_SEND", FALSOCK_ERR_SEND );
   this->addConstant( "ERR_UNRESOLVED", FALSOCK_ERR_UNRESOLVED );
   this->addConstant( "ERR_ALREADY_CREATED", FALSOCK_ERR_ALREADY_CREATED );

#if WITH_OPENSSL
   this->addConstant( "ERR_SSLCONFIG", FALSOCK_ERR_SSLCONFIG );
   this->addConstant( "ERR_SSLCONNECT", FALSOCK_ERR_SSLCONNECT );
#endif

   this->addConstant("SOCK_NONE", (int64) -1 );
   this->addConstant("SOCK_DGRAM", (int64) SOCK_DGRAM );
   this->addConstant("SOCK_STREAM", (int64) SOCK_STREAM );
   this->addConstant("SOCK_RAW", (int64) SOCK_RAW );
#ifdef SOCK_SEQPACKET
   this->addConstant("SOCK_SEQPACKET", (int64) SOCK_SEQPACKET );
#endif

   this->addConstant("AF_INET", (int64) AF_INET );
   this->addConstant("AF_INET6", (int64) AF_INET6 );
   this->addConstant("AF_IPX", (int64) AF_IPX );
   this->addConstant("AF_UNIX", (int64) AF_UNIX );
#ifdef AF_LOCAL
   this->addConstant("AF_LOCAL", (int64) AF_LOCAL );
#endif

   this->addConstant("SO_DEBUG", (int64) SO_DEBUG );
   this->addConstant("SO_REUSEADDR", (int64) SO_REUSEADDR );
   this->addConstant("SO_TYPE", (int64) SO_TYPE );
   this->addConstant("SO_ERROR", (int64) SO_ERROR );
   this->addConstant("SO_DONTROUTE", (int64) SO_DONTROUTE );
   this->addConstant("SO_BROADCAST", (int64) SO_BROADCAST );
   this->addConstant("SO_KEEPALIVE", (int64) SO_KEEPALIVE );
   this->addConstant("SO_RCVBUF", (int64) SO_RCVBUF );
   this->addConstant("SO_OOBINLINE", (int64) SO_OOBINLINE );
   this->addConstant("SO_LINGER", (int64) SO_LINGER );
   this->addConstant("SO_RCVLOWAT", (int64) SO_RCVLOWAT );
   this->addConstant("SO_SNDLOWAT", (int64) SO_SNDLOWAT );
   this->addConstant("SO_RCVTIMEO", (int64) SO_RCVTIMEO );
   this->addConstant("SO_SNDTIMEO", (int64) SO_SNDTIMEO );

   this->addConstant("SOL_SOCKET",  (int64) SOL_SOCKET );
   this->addConstant("IPPROTO_IP",  (int64) IPPROTO_IP );
#ifdef IPPROTO_IPV6
   this->addConstant("IPPROTO_IPV6", (int64) IPPROTO_IPV6 );
#endif
#ifdef IPPROTO_RAW
   this->addConstant("IPPROTO_RAW", (int64) IPPROTO_RAW );
#endif
   this->addConstant("IPPROTO_TCP", (int64) IPPROTO_TCP );
   this->addConstant("IPPROTO_UDP", (int64) IPPROTO_UDP );

#if WITH_OPENSSL
#ifndef OPENSSL_NO_SSL2
   this->addConstant("SSLv2", (int64) Mod::SSLData::SSLv2 );
#endif
   this->addConstant("SSLv23", (int64) Mod::SSLData::SSLv23 );
   this->addConstant("SSLv3", (int64) Mod::SSLData::SSLv3 );
   this->addConstant("TLSv1", (int64) Mod::SSLData::TLSv1 );
   this->addConstant("DTLSv1", (int64) Mod::SSLData::DTLSv1 );


   Mod::ssl_init();

#endif
}

ModuleInet::~ModuleInet()
{
#if WITH_OPENSSL
   Mod::ssl_fini();
#endif
#ifdef FALCON_SYS_WINDOWS
   delete m_smpxf;
#endif
   Mod::shutdown_system();
}

}
}

/* end of socket_ext.cpp */
