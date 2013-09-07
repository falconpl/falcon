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
#define FALSOCK_ERR_GENERIC_MSG "Generic network error"
#define FALSOCK_ERR_RESOLV  (FALCON_SOCKET_ERROR_BASE + 1)
#define FALSOCK_ERR_RESOLV_MSG "Network error during address resolution"
#define FALSOCK_ERR_CREATE  (FALCON_SOCKET_ERROR_BASE + 2)
#define FALSOCK_ERR_CREATE_MSG "Network error creating a socket"
#define FALSOCK_ERR_CONNECT  (FALCON_SOCKET_ERROR_BASE + 3)
#define FALSOCK_ERR_CONNECT_MSG "Network error during connection"
#define FALSOCK_ERR_SEND  (FALCON_SOCKET_ERROR_BASE + 4)
#define FALSOCK_ERR_SEND_MSG "Network error during send"
#define FALSOCK_ERR_RECV  (FALCON_SOCKET_ERROR_BASE + 5)
#define FALSOCK_ERR_RECV_MSG "Network error during receive"
#define FALSOCK_ERR_CLOSE  (FALCON_SOCKET_ERROR_BASE + 6)
#define FALSOCK_ERR_CLOSE_MSG "Network error during close"
#define FALSOCK_ERR_BIND  (FALCON_SOCKET_ERROR_BASE + 7)
#define FALSOCK_ERR_BIND_MSG "Network error during bind"
#define FALSOCK_ERR_ACCEPT  (FALCON_SOCKET_ERROR_BASE + 8)
#define FALSOCK_ERR_ACCEPT_MSG "Network error during accept"

#define FALSOCK_ERR_INCOMPATIBLE       (FALCON_SOCKET_ERROR_BASE + 9)
#define FALSOCK_ERR_INCOMPATIBLE_MSG   "Socket already configured"

#define FALSOCK_ERR_UNRESOLVED         (FALCON_SOCKET_ERROR_BASE + 10)
#define FALSOCK_ERR_UNRESOLVED_MSG     "Unresolved address used in operation"

#define FALSOCK_ERR_FCNTL              (FALCON_SOCKET_ERROR_BASE + 11)
#define FALSOCK_ERR_FCNTL_MSG          "Error in FCNTL set/get"

#define FALSOCK_ERR_LISTEN             (FALCON_SOCKET_ERROR_BASE + 12)
#define FALSOCK_ERR_LISTEN_MSG         "Network error during listen"

#if WITH_OPENSSL
#define FALSOCK_ERR_SSLCONFIG (FALCON_SOCKET_ERROR_BASE + 15)
#define FALSOCK_ERR_SSLCONNECT (FALCON_SOCKET_ERROR_BASE + 16)
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

FALCON_DECLARE_ERROR( NetError );

}
}

#endif

/* end of socket_ext.h */
