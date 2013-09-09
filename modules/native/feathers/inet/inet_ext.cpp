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

#define SRC "modules/native/feathres/inet/inet_ext.cpp"

#include <falcon/autocstring.h>
#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>

#include "inet_ext.h"
#include "inet_mod.h"

#include <errno.h>

/*#
   @beginmodule feathers.socket
*/

namespace Falcon {

static Mod::Address* internal_read_address( Function* func, Item* i_address, bool& made )
{
   static Class* clsAddress = static_cast<Ext::ModuleInet*>(func->fullModule())->addressClass();

   Mod::Address* target = 0;
   made = false;
   if( i_address != 0 )
   {
      if( i_address->isString() )
      {
         const String& sTarget = *i_address->asString();
         // this might throw
         target = new Mod::Address(sTarget);
         made = true;
      }
      else
      {
         // might return 0
         target = static_cast<Mod::Address*>(i_address->asParentInst( clsAddress ));
      }
   }

   if( target == 0 )
   {
      throw func->paramError(__LINE__, SRC);
   }

   return target;
}

namespace Ext {

namespace {
/*#
   @function getHostName
   @brief Retreives the host name of the local machine.
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
   @optparam type Socket type.
   @brief IP networking base class.
   @raise NetError if the socket creation fails.

   The Socket class implements a system level, network-oriented
   socket implementation.

   The default type of the socket is  STREAM, suitable to create
   TCP/IP client and server sockets. Available types are:

   - Socket.DGRAM: Creates a datagram-oriented (UDP) socket.
   - Socket.STREAM: Creates a stateful connection oriented (TCP) socket.
   - Socket.RAW: Creates a raw IP socket.
   - Socket.SEQPACKET: Available on some POSIX systems only, creates
                       a reliable sequence datagram-oriented protocol.
                       Where not available, this constant assumes the
                       value of -1
*/

/*#
@property address Socket
@brief Return the address associated with a socket.
*/
void get_address(const Class* cls, const String&, void* instance, Item& value )
{
   TRACE1( "Socket.address for %p", instance );
   ModuleInet* inet = static_cast<ModuleInet*>( cls->module() );
   Mod::Socket* socket = static_cast<Mod::Socket*>(instance);
   Mod::Address* a = socket->address();
   a->incref();
   value = FALCON_GC_STORE(inet->addressClass(), a);
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

   int family = i_family == 0 || i_family->isNil() ? AF_INET6 : (int) i_family->forceInteger();
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
   in @a Socket.setTimeout method. In that case, the @a TCPSocket.isConnected method may check if the
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
   bool made = false;
   Mod::Address* target = internal_read_address(this, ctx->param(0), made);
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
      // decref if we created the target address.
      if( made )
      {
         target->decref();
      }
   }
   catch( ... )
   {
      if( made )
      {
         target->decref();
      }
      throw;
   }

   ctx->returnFrame();
}


/*#
   @method isConnected TCPSocket
   @brief Check if this TCPSocket is currently connected with a remote host.
   @return True if the socket is currently connected, false otherwise.
   @raise NetError if a the connection process caused an error in the meanwhile.

   This method checks if this TCPSocket is currently connected with a remote host.

   @see TCPSocket.connect
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
   @method send TCPSocket
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

   This method works as the TCPSocket.send method, with the
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
   bool made = false;
   Mod::Address* address = internal_read_address(this, ctx->param(0), made);

   try {
      internal_send( this, ctx, address );
      if( made ) {
         address->decref();
      }
   }
   catch( ... )
   {
      if( made ) {
         address->decref();
      }
   }
}



/*#
   @method recv TCPSocket
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

   buffer->clear();
   length_t size = (length_t) i_size->forceInteger();
   buffer->reserve( size );
   buffer->toMemBuf();

   int64 res = (int64) sock->recv( buffer->getRawStorage(), size, 0 );
   buffer->size(res);
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
   buffer->size(res);
   ctx->returnFrame( res );
}

/*#
   @method closeRead TCPSocket
   @brief Closes a socket read side.
   @return False if timed out, true if successful
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
FALCON_DECLARE_FUNCTION(closeWrite, "")
FALCON_DEFINE_FUNCTION_P1(closeWrite)
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());
   sock->closeWrite();
   ctx->returnFrame();
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
FALCON_DECLARE_FUNCTION(close, "")
FALCON_DEFINE_FUNCTION_P1(close)
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(ctx->self().asInst());
   sock->close();
   ctx->returnFrame();
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


void set_broadcasting(const Class*, const String&, void* instance, Item& value )
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


void set_nonblocking(const Class*, const String&, void* instance, Item& value )
{
   Mod::Socket* sock = static_cast<Mod::Socket*>(instance);
   sock->setNonBlocking(value.isTrue());
}


/*#
   @method bind Socket
   @brief Specify the address and port at which this server will be listening.
   @param address Address at which this server will be listening.
   @optparam dgram Sets the socket to be a datagram-oriented socket.
   @raise NetError on system error.

   This method binds the socket to an address, possibly preparing the socket
   to send and receive datagram packets via the UDP protocol


   In case the system cannot bind the required address, a NetError is raised.
   After a successful @b bind call, the socket might be further configured with
   access to the @a Socket.broadcasting property, or via @a Socket.connect.

   If the socket is set as datagram, Socket.connect will have the effect of setting a
   default target address for the socket, so that every @a Socket.send operation
   will be addressed to that remote interface.
*/
FALCON_FUNC  TCPServer_bind(  )
{

}

#if WITH_OPENSSL
/*#
   @method sslConfig TCPSocket
   @brief Prepare a socket for SSL operations.
   @param serverSide True for a server-side socket, false for a client-side socket.
   @optparam version SSL method (one of SSLv2, SSLv3, SSLv23, TLSv1, DTLSv1 ). Default SSLv3
   @optparam cert Certificate file
   @optparam pkey Private key file
   @optparam ca Certificate authorities file

   Must be called after socket is really created, that is after connect() is called.
 */
FALCON_FUNC TCPSocket_sslConfig( ::Falcon::VMachine* vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   Item* i_asServer = vm->param( 0 );
   Item* i_sslVer = vm->param( 1 );
   Item* i_cert = vm->param( 2 );
   Item* i_pkey = vm->param( 3 );
   Item* i_ca = vm->param( 4 );

   if ( !i_asServer || !( i_asServer->isBoolean() )
      || !i_sslVer || !( i_sslVer->isInteger() )
      || ( i_cert && !( i_cert->isString() || i_cert->isNil() ) )
      || ( i_pkey && !( i_pkey->isString() || i_pkey->isNil() ) )
      || ( i_ca && !( i_ca->isString() || i_ca->isNil() ) ) )
   {
      throw  new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "B,I,[S,S,S]" ) );
   }

   AutoCString s_cert( "" );
   if ( i_cert && !i_cert->isNil() )
   {
      s_cert.set( i_cert->asString() );
   }
   AutoCString s_pkey( "" );
   if ( i_pkey && !i_pkey->isNil() )
   {
      s_pkey.set( i_pkey->asString() );
   }
   AutoCString s_ca( "" );
   if ( i_ca && !i_ca->isNil() )
   {
      s_ca.set( i_ca->asString() );
   }

   Sys::SSLData::ssl_error_t res = tcps->sslConfig( i_asServer->asBoolean(),
                        (Sys::SSLData::sslVersion_t) i_sslVer->asInteger(),
                        s_cert.c_str(), s_pkey.c_str(), s_ca.c_str() );

   if ( res != Sys::SSLData::no_error )
   {
      throw new NetError( ErrorParam( FALSOCK_ERR_SSLCONFIG, __LINE__ )
         .desc( FAL_STR( sk_msg_errsslconfig ) )
         .sysError( (uint32) res ) );
   }
}


/*#
   @method sslConnect TCPSocket
   @brief Negotiate an SSL connection.

   Must be called after socket is connected and has been properly configured for
   SSL operations.
 */
FALCON_FUNC TCPSocket_sslConnect( ::Falcon::VMachine* vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   Sys::SSLData::ssl_error_t res = tcps->sslConnect();

   if ( res != Sys::SSLData::no_error )
   {
      throw new NetError( ErrorParam( FALSOCK_ERR_SSLCONNECT, __LINE__ )
         .desc( FAL_STR( sk_msg_errsslconnect ) )
         .sysError( (uint32) tcps->lastError() ) );
   }
}


/*#
   @method sslClear TCPSocket
   @brief Free resources taken by SSL contexts.

   Useful if you want to reuse a socket.
 */
FALCON_FUNC TCPSocket_sslClear( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::TCPSocket *tcps = (Sys::TCPSocket *) self->getUserData();

   tcps->sslClear();
   vm->retnil();
}
#endif // WITH_OPENSSL

} /* NS CSocket */

ClassSocket::ClassSocket():
     Class("Socket")
{
   this->m_bHasSharedInstances = true;

   setConstuctor( new CSocket::Function_init );

   addConstant("NONE", (int64) -1 );
   addConstant("DGRAM", (int64) SOCK_DGRAM );
   addConstant("STREAM", (int64) SOCK_STREAM );
   addConstant("RAW", (int64) SOCK_RAW );
#ifdef SOCK_SEQPACKET
   addConstant("SEQPACKET", (int64) SOCK_SEQPACKET );
#endif

   addProperty("address",&CSocket::get_address);
   addProperty("descriptor",&CSocket::get_descriptor);
   addProperty("host",&CSocket::get_host);
   addProperty("service",&CSocket::get_service);
   addProperty("port",&CSocket::get_port);
   addProperty("type",&CSocket::get_type);

   addMethod( new CSocket::Function_connect );
   addMethod( new CSocket::Function_isConnected );

   addMethod( new CSocket::Function_recv );
   addMethod( new CSocket::Function_recvFrom );
   addMethod( new CSocket::Function_send );
   addMethod( new CSocket::Function_sendTo );
   addMethod( new CSocket::Function_closeRead );
   addMethod( new CSocket::Function_closeWrite );
   addMethod( new CSocket::Function_close );

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
   @return A new TCPSocket after a successful connection.
   @raise NetError on system error.

   This method accepts incoming connection and creates a TCPSocket object that
   can be used to communicate with the remote host. Before calling accept(), it is
   necessary to have successfully called bind() to bind the listening application to
   a certain local address.

   If a timeout is not specified, the function will block until a TCP connection is
   received. If it is specified, is a number of millisecond that will be waited
   before returning a nil. Setting the timeout to zero will cause accept to return
   immediately, providing a valid TCPSocket as return value only if an incoming
   connection was already pending.

   The wait blocks the VM, and thus, also the other coroutines.
   If a system error occurs during the wait, a NetError is raised.
*/
FALCON_FUNC  TCPServer_accept(  )
{

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
            *res = "[" + host + "]:" + service;
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


//=================================================================
// Module
//=================================================================

ModuleInet::ModuleInet():
         Module("inet")
{
   m_clsAddress = new ClassAddress;
   m_clsSocket = new ClassSocket;
   m_clsResolver = new ClassResolver;


   *this
      << m_clsAddress
      << m_clsSocket
      << m_clsResolver
      << new ClassNetError
      ;

   //====================================
   // Net error code enumeration

   /*#
      @module inet
      @brief Network failure error categories.

      This error codes define macro-categories of network errors that
      have appened. Details are available by reading the system specific
      net-error.

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

}

ModuleInet::~ModuleInet()
{
   Mod::shutdown_system();
}

}
}

/* end of socket_ext.cpp */
