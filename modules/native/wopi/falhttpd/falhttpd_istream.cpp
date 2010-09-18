/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_istream.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Servicing single client.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 28 Feb 2010 23:03:18 +0100
s
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#include "falhttpd_istream.h"

#include "falhttpd_istream.h"

#ifndef FALCON_SYSTEM_WIN
#include <errno.h>
#include <unistd.h>
#endif

SocketInputStream::SocketInputStream( SOCKET s ):
   Stream(t_network),
   m_socket(s),
   m_nLastError(0)
{
   m_status = t_open;
}

SocketInputStream::~SocketInputStream()
{
   // don't do anything
}

SocketInputStream* SocketInputStream::clone() const
{
   // uncloneable;
   return 0;
}

Falcon::int32 SocketInputStream::read( void *buffer, Falcon::int32 size )
{
   int rin = ::recv( m_socket, (char*) buffer, size, 0 );

   if( rin == 0 )
   {
      status(t_eof);
   }
   else if( rin < 0 )
   {
      status(t_error);
      #ifdef FALCON_SYSTEM_WINDOWS
      m_nLastError =  (Falcon::int64) ::WSAGetLastError();
      #else
      m_nLastError = (Falcon::int64) errno;
      #endif
      return -1;
   }

   return (Falcon::int32) rin;
}

bool SocketInputStream::get( Falcon::uint32 &chr )
{
   // not efficient, but we're not actually going to use this.
   Falcon::byte b;
   return this->read( &b, 1 ) == 1;
}


Falcon::int64 SocketInputStream::lastError() const
{
   return m_nLastError;
}

/* end of falhttpd_istream.cpp */

