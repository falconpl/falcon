/*
   FALCON - The Falcon Programming Language.
   FILE: socket.cpp

   The socket module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The socket module.
*/

#include <falcon/module.h>
#include "socket_ext.h"
#include "socket_sys.h"
#include "socket_st.h"

#include "version.h"

/*#
   @module feathers.socket Low level IP networking.
   @brief Low level TCP/IP networking support.

   The socket module provides a low level access to network (TCP/IP) socket wise
   functions. UDP and TCP protocol are supported, and the module provides also name
   resolution facilities, both performed automatically when calling connect and
   bind methods, or manually by calling an appropriate name or address resolution
   routine.

   The module supports both IPv4 and IPv6 networking; generally, IPv6 is chosen
   transparently when an IPv6 address is provided or retrieved by the name
   resolution system, if the host system supports it.

   The Socket module defines a @a NetError class that is raised on network errors. The
   class is derived from core IoError and doesn't add any method or property.

   @note The module can be loaded using the command
      @code
      load socket
      @endcode

   @beginmodule feathers.socket
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   if ( ! Falcon::Sys::init_system() )
   {
      return 0;
   }

   Falcon::Module *self = new Falcon::Module();
   self->name( "socket" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "socket_st.h"

   //====================================
   // Net error code enumeration

   /*#
      @enum NetErrorCode
      @brief Network failure error categories.

      This error codes define macro-categories of network errors that
      have appened. Details are available by reading the system specific
      net-error.

      The @a NetError.code property assumes one of this values:

      - @b generic: A generic failure prevented the network layer to work
                     altogether. I.e. it was not possible to initialize
                     the network layer
      - @b resolv: An error happened while trying to resolve a network
                  address; possibly, the name resolution service was
                  not available or failed altogether.
      - @b create: It was impossible to create the socket.
      - @b send: The network had an error while trying to send data.
      - @b receive: The network had an error while receiving data from a remote host.
      - @b close: An error was detected while closing the socket. Either the socket
                  could not be closed (i.e. because it was already invalid) or the
                  close sequence was disrupted by a network failure.
      - @b bind: The required address could not be allocated by the calling process.
                 Either the address is already busy or the bind operation required
                 privileges not owned by the process.
      - @b accept: The network system failed while accepting an incoming connection.
                 This usually means that the accepting thread has become unavailable.
   */
   Falcon::Symbol *c_errcode = self->addClass( "NetErrorCode" );
   self->addClassProperty( c_errcode, "generic").setInteger( FALSOCK_ERR_GENERIC );
   self->addClassProperty( c_errcode, "resolv").setInteger( FALSOCK_ERR_RESOLV );
   self->addClassProperty( c_errcode, "create").setInteger( FALSOCK_ERR_CREATE );
   self->addClassProperty( c_errcode, "send").setInteger( FALSOCK_ERR_SEND );
   self->addClassProperty( c_errcode, "receive").setInteger( FALSOCK_ERR_RECV );
   self->addClassProperty( c_errcode, "close").setInteger( FALSOCK_ERR_CLOSE );
   self->addClassProperty( c_errcode, "bind").setInteger( FALSOCK_ERR_BIND );
   self->addClassProperty( c_errcode, "accept").setInteger( FALSOCK_ERR_ACCEPT );


   //====================================
   // Generic functions

   self->addExtFunc( "getHostName", Falcon::Ext::falcon_getHostName );
   self->addExtFunc( "resolveAddress", Falcon::Ext::resolveAddress )->
      addParam("address");
   self->addExtFunc( "socketErrorDesc", Falcon::Ext::socketErrorDesc )->
      addParam("code");

   // private class socket.
   Falcon::Symbol *c_socket = self->addClass( "Socket", Falcon::Ext::Socket_init, false );
   self->addClassMethod( c_socket, "getTimeout", Falcon::Ext::Socket_getTimeout );
   self->addClassMethod( c_socket, "setTimeout", Falcon::Ext::Socket_setTimeout ).asSymbol()->
      addParam("timeout");
   self->addClassMethod( c_socket, "dispose", Falcon::Ext::Socket_dispose );
   self->addClassMethod( c_socket, "readAvailable", Falcon::Ext::Socket_readAvailable ).asSymbol()->
      addParam("timeout");
   self->addClassMethod( c_socket, "writeAvailable", Falcon::Ext::Socket_writeAvailable ).asSymbol()->
      addParam("timeout");
   self->addClassMethod( c_socket, "getService", Falcon::Ext::Socket_getService );
   self->addClassMethod( c_socket, "getHost", Falcon::Ext::Socket_getHost );
   self->addClassMethod( c_socket, "getPort", Falcon::Ext::Socket_getPort );
   self->addClassProperty( c_socket, "timedOut" );
   self->addClassProperty( c_socket, "lastError" );

   Falcon::Symbol *tcpsocket = self->addClass( "TCPSocket", Falcon::Ext::TCPSocket_init );
   tcpsocket->setWKS( true ); // needed by TCPServer
   tcpsocket->getClassDef()->addInheritance(  new Falcon::InheritDef( c_socket ) );
   self->addClassMethod( tcpsocket, "connect", Falcon::Ext::TCPSocket_connect ).asSymbol()->
      addParam("host")->addParam("service");
   self->addClassMethod( tcpsocket, "isConnected", Falcon::Ext::TCPSocket_isConnected );
   self->addClassMethod( tcpsocket, "send", Falcon::Ext::TCPSocket_send ).asSymbol()->
      addParam("buffer")->addParam("size")->addParam("start");
   self->addClassMethod( tcpsocket, "recv", Falcon::Ext::TCPSocket_recv ).asSymbol()->
      addParam("bufOrSize");
   self->addClassMethod( tcpsocket, "close", Falcon::Ext::TCPSocket_close );
   self->addClassMethod( tcpsocket, "closeRead", Falcon::Ext::TCPSocket_closeRead );
   self->addClassMethod( tcpsocket, "closeWrite", Falcon::Ext::TCPSocket_closeWrite );

   Falcon::Symbol *udpsocket = self->addClass( "UDPSocket", Falcon::Ext::UDPSocket_init );
   udpsocket->getClassDef()->addInheritance(  new Falcon::InheritDef( c_socket ) );
   self->addClassMethod( udpsocket, "broadcast", Falcon::Ext::UDPSocket_broadcast );
   self->addClassMethod( udpsocket, "sendTo", Falcon::Ext::UDPSocket_sendTo ).asSymbol()->
      addParam("host")->addParam("service")->addParam("buffer")->addParam("size")->addParam("start");
   self->addClassMethod( udpsocket, "recv", Falcon::Ext::UDPSocket_recv ).asSymbol()->
      addParam("bufOrSize");
   self->addClassProperty( udpsocket, "remote" );
   self->addClassProperty( udpsocket, "remoteService" );

   Falcon::Symbol *tcpserver = self->addClass( "TCPServer", Falcon::Ext::TCPServer_init );
   self->addClassMethod( tcpserver, "dispose", Falcon::Ext::TCPServer_dispose );
   self->addClassMethod( tcpserver, "bind", Falcon::Ext::TCPServer_bind ).asSymbol()->
      addParam("addrOrService")->addParam("service");
   self->addClassMethod( tcpserver, "accept", Falcon::Ext::TCPServer_accept ).asSymbol()->
      addParam("timeout");
   self->addClassProperty( tcpserver, "lastError" );

   //==================================================
   // Error class

   Falcon::Symbol *error_class = self->addExternalRef( "IoError" ); // it's external
   Falcon::Symbol *neterr_cls = self->addClass( "NetError", Falcon::Ext::NetError_init );
   neterr_cls->setWKS( true );
   neterr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );


   return self;
}

namespace Falcon{
namespace Sys{

Socket::Socket( const Socket& other ):
   m_address( other.m_address ),
   d(other.d),
   m_ipv6( other.m_ipv6 ),
   m_lastError(other.m_lastError),
   m_timeout(other.m_timeout),
   m_boundFamily(other.m_boundFamily),
   m_refcount(other.m_refcount)
{
   atomicInc( *other.m_refcount );
}

FalconData *Socket::clone() const
{
   return new Socket(*this);
}

Socket::~Socket()
{
   if( atomicDec( *m_refcount ) == 0 )
   {      
      // ungraceful close.
      terminate();
      memFree( (void*)m_refcount );
   }
}


TCPSocket::TCPSocket( const TCPSocket& other ):
   Socket( other ),
   m_connected(other.m_connected)
{
}


FalconData* TCPSocket::clone() const
{
   return new TCPSocket(*this);
}
}
}

/* end of socket.cpp */

