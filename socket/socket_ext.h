/*
   FALCON - The Falcon Programming Language.
   FILE: socket_ext.cpp
   $Id: socket_ext.h,v 1.2 2007/03/04 17:39:03 jonnymind Exp $

   Falcon VM interface to socket module -- header.
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
   Falcon VM interface to socket module -- header.
*/


#ifndef FLC_SOCKET_EXT_H
#define FLC_SOCKET_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

// =============================================
// Generic Functions
// ==============================================
FALCON_FUNC  falcon_getHostName( ::Falcon::VMachine *vm );
FALCON_FUNC  resolveAddress( ::Falcon::VMachine *vm );
FALCON_FUNC  socketErrorDesc( ::Falcon::VMachine *vm );

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
