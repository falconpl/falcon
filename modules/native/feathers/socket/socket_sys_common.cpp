/*
   FALCON - The Falcon Programming Language.
   FILE: socket_sys_common.cpp

   The socket module (system independant methods).
   -------------------------------------------------------------------
   Author: Stanislas Marquis
   Begin: 2011-11-05 15:59:13

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file socket_sys_common.cpp
 *  The socket module (system independant methods).
 */

#include <falcon/module.h>
#include "socket_ext.h"
#include "socket_sys.h"
#include "socket_st.h"

namespace Falcon {
namespace Sys {

// ====================================================
// private class socket.

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
      free( (void*)m_refcount );
   }
}

// =================================================
// TCP socket.

TCPSocket::TCPSocket( const TCPSocket& other ):
   Socket( other ),
   m_connected(other.m_connected)
{
}


FalconData* TCPSocket::clone() const
{
   return new TCPSocket(*this);
}

} // namespace Sys
} // namespace Falcon

/* end of socket_sys_common.cpp */
/* vim: set ai et sw=3 ts= sts=3: */
