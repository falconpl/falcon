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

#include "version.h"

FALCON_MODULE_DECL(const Falcon::EngineData &data )
{
   data.set();

   if ( ! Falcon::Sys::init_system() )
   {
      return 0;
   }

   Falcon::Module *self = new Falcon::Module();
   self->name( "socket" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   self->addExtFunc( "getHostName", Falcon::Ext::falcon_getHostName );
   self->addExtFunc( "resolveAddress", Falcon::Ext::resolveAddress );
   self->addExtFunc( "socketErrorDesc", Falcon::Ext::socketErrorDesc );

   // private class socket.
   Falcon::Symbol *c_socket = self->addClass( "Socket", Falcon::Ext::Socket_init, false );
   self->addClassMethod( c_socket, "getTimeout", Falcon::Ext::Socket_getTimeout );
   self->addClassMethod( c_socket, "setTimeout", Falcon::Ext::Socket_setTimeout );
   self->addClassMethod( c_socket, "dispose", Falcon::Ext::Socket_dispose );
   self->addClassMethod( c_socket, "readAvailable", Falcon::Ext::Socket_readAvailable );
   self->addClassMethod( c_socket, "writeAvailable", Falcon::Ext::Socket_writeAvailable );
   self->addClassMethod( c_socket, "getService", Falcon::Ext::Socket_getService );
   self->addClassMethod( c_socket, "getHost", Falcon::Ext::Socket_getHost );
   self->addClassMethod( c_socket, "getPort", Falcon::Ext::Socket_getPort );
   self->addClassProperty( c_socket, "timedOut" );
   self->addClassProperty( c_socket, "lastError" );

   Falcon::Symbol *tcpsocket = self->addClass( "TCPSocket", Falcon::Ext::TCPSocket_init );
   tcpsocket->setWKS( true ); // needed by TCPServer
   tcpsocket->getClassDef()->addInheritance(  new Falcon::InheritDef( c_socket ) );
   self->addClassMethod( tcpsocket, "connect", Falcon::Ext::TCPSocket_connect );
   self->addClassMethod( tcpsocket, "isConnected", Falcon::Ext::TCPSocket_isConnected );
   self->addClassMethod( tcpsocket, "send", Falcon::Ext::TCPSocket_send );
   self->addClassMethod( tcpsocket, "recv", Falcon::Ext::TCPSocket_recv );
   self->addClassMethod( tcpsocket, "close", Falcon::Ext::TCPSocket_close );
   self->addClassMethod( tcpsocket, "closeRead", Falcon::Ext::TCPSocket_closeRead );
   self->addClassMethod( tcpsocket, "closeWrite", Falcon::Ext::TCPSocket_closeWrite );

   Falcon::Symbol *udpsocket = self->addClass( "UDPSocket", Falcon::Ext::UDPSocket_init );
   udpsocket->getClassDef()->addInheritance(  new Falcon::InheritDef( c_socket ) );
   self->addClassMethod( udpsocket, "broadcast", Falcon::Ext::UDPSocket_broadcast );
   self->addClassMethod( udpsocket, "sendTo", Falcon::Ext::UDPSocket_sendTo );
   self->addClassMethod( udpsocket, "recv", Falcon::Ext::UDPSocket_recv );
   self->addClassProperty( udpsocket, "remote" );
   self->addClassProperty( udpsocket, "remoteService" );

   Falcon::Symbol *tcpserver = self->addClass( "TCPServer", Falcon::Ext::TCPServer_init );
   self->addClassMethod( tcpserver, "dispose", Falcon::Ext::TCPServer_dispose );
   self->addClassMethod( tcpserver, "bind", Falcon::Ext::TCPServer_bind );
   self->addClassMethod( tcpserver, "accept", Falcon::Ext::TCPServer_accept );
   self->addClassProperty( tcpserver, "lastError" );

   //==================================================
   // Error class

   Falcon::Symbol *error_class = self->addExternalRef( "IoError" ); // it's external
   Falcon::Symbol *neterr_cls = self->addClass( "NetError", Falcon::Ext::NetError_init );
   neterr_cls->setWKS( true );
   neterr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );


   return self;
}

/* end of socket.cpp */
