/*
   FALCON - The Falcon Programming Language.
   FILE: socket_ext.cpp

   Falcon VM interface to socket module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon VM interface to socket module -- header.
*/


#ifndef FLC_SOCKET_EXT_H
#define FLC_SOCKET_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/error_base.h>

#ifndef FALCON_SOCKET_ERROR_BASE
   #define FALCON_SOCKET_ERROR_BASE        1170
#endif

#define FALSOCK_ERR_GENERIC  (FALCON_SOCKET_ERROR_BASE + 0)
#define FALSOCK_ERR_RESOLV  (FALCON_SOCKET_ERROR_BASE + 1)
#define FALSOCK_ERR_CREATE  (FALCON_SOCKET_ERROR_BASE + 2)
#define FALSOCK_ERR_CONNECT  (FALCON_SOCKET_ERROR_BASE + 3)
#define FALSOCK_ERR_SEND  (FALCON_SOCKET_ERROR_BASE + 4)
#define FALSOCK_ERR_RECV  (FALCON_SOCKET_ERROR_BASE + 5)
#define FALSOCK_ERR_CLOSE  (FALCON_SOCKET_ERROR_BASE + 6)
#define FALSOCK_ERR_BIND  (FALCON_SOCKET_ERROR_BASE + 7)
#define FALSOCK_ERR_ACCEPT  (FALCON_SOCKET_ERROR_BASE + 8)

#if WITH_OPENSSL
#define FALSOCK_ERR_SSLCONFIG (FALCON_SOCKET_ERROR_BASE + 10)
#define FALSOCK_ERR_SSLCONNECT (FALCON_SOCKET_ERROR_BASE + 11)
#endif

namespace Falcon {
namespace Ext {

// =============================================
// Generic Functions
// ==============================================
FALCON_FUNC  falcon_getHostName( ::Falcon::VMachine *vm );
FALCON_FUNC  resolveAddress( ::Falcon::VMachine *vm );
FALCON_FUNC  socketErrorDesc( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_haveSSL( ::Falcon::VMachine *vm );

// ==============================================
// Class Socket
// ==============================================
FALCON_FUNC  Socket_init( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_setTimeout( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_getTimeout( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_dispose( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_readAvailable( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_writeAvailable( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_getHost( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_getService( ::Falcon::VMachine *vm );
FALCON_FUNC  Socket_getPort( ::Falcon::VMachine *vm );

// ==============================================
// Class TCPSocket
// ==============================================
FALCON_FUNC  TCPSocket_init( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_connect( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_isConnected( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_send( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_recv( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_closeRead( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_closeWrite( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_close( ::Falcon::VMachine *vm );
#if WITH_OPENSSL
FALCON_FUNC  TCPSocket_sslConfig( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_sslClear( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_sslConnect( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_sslRead( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPSocket_sslWrite( ::Falcon::VMachine *vm );
#endif

// ==============================================
// Class UDPSocket
// ==============================================
FALCON_FUNC  UDPSocket_init( ::Falcon::VMachine *vm );
FALCON_FUNC  UDPSocket_broadcast( ::Falcon::VMachine *vm );
FALCON_FUNC  UDPSocket_sendTo( ::Falcon::VMachine *vm );
FALCON_FUNC  UDPSocket_recv( ::Falcon::VMachine *vm );

// ==============================================
// Class TCPServer
// ==============================================

FALCON_FUNC  TCPServer_init( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPServer_dispose( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPServer_bind( ::Falcon::VMachine *vm );
FALCON_FUNC  TCPServer_accept( ::Falcon::VMachine *vm );

class NetError: public ::Falcon::Error
{
public:
   NetError():
      Error( "NetError" )
   {}

   NetError( const ErrorParam &params  ):
      Error( "NetError", params )
      {}
};

FALCON_FUNC  NetError_init ( ::Falcon::VMachine *vm );

}
}

#endif

/* end of socket_ext.h */
