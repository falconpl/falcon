/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_ostream.h

   Micro HTTPD server providing Falcon scripts on the web.

   Output stream with reply-header control.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Mar 2010 20:48:45 +0100
s
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "falhttpd_ostream.h"
#include "falhttpd_reply.h"

class FalhttpdReply;

FalhttpdOutputStream::FalhttpdOutputStream( SOCKET s ):
      Stream(t_network),
      m_socket( s )
{
   m_status = t_open;
}


FalhttpdOutputStream::~FalhttpdOutputStream()
{
   // do nothing
}


FalhttpdOutputStream* FalhttpdOutputStream::clone() const
{
   return new FalhttpdOutputStream( m_socket );
}


bool FalhttpdOutputStream::get( Falcon::uint32 &chr )
{
   chr = 0;
   return false;
}

Falcon::int32 FalhttpdOutputStream::write( const void *buffer, Falcon::int32 size )
{
   int sent = 0;
   const char* cbuf = (const char*) buffer;
   while( sent < size )
   {
      int res = ::send( m_socket, cbuf + sent, size - sent, 0 );
      if( res < 0 )
      {
         return res;
      }
      sent += res;
      m_lastMoved = sent;
   }

   return sent;
}


bool FalhttpdOutputStream::put( Falcon::uint32 chr )
{
   Falcon::byte b = (Falcon::byte) chr;
   return ::send( m_socket, (const char*) &b, 1, 0 ) == 1;
}


Falcon::int64 FalhttpdOutputStream::lastError() const
{
   return m_nLastError;
}

/* falhttpd_ostream.cpp */
