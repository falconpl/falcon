/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_reply.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Reply object specific for the Falcon micro HTTPD server.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Mar 2010 21:31:37 +0100
s
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "falhttpd_reply.h"
#include "falhttpd_ostream.h"

FalhttpdReply::FalhttpdReply( const Falcon::CoreClass* cls ):
   Reply( cls )
{
}

FalhttpdReply::~FalhttpdReply()
{
}


void FalhttpdReply::init( SOCKET s )
{
   m_socket = s;
}


void FalhttpdReply::startCommit()
{
   m_headers.A("HTTP/1.1 ").N( m_nStatus ).A(" ").A( m_sReason ).A("\r\n");
}


Falcon::Stream* FalhttpdReply::makeOutputStream()
{
   return new Falcon::StreamBuffer( new FalhttpdOutputStream( m_socket ) );
}

void FalhttpdReply::commitHeader( const Falcon::String& hname, const Falcon::String& hvalue )
{
   m_headers += hname + ": " + hvalue + "\r\n";
}

void FalhttpdReply::endCommit()
{
   send( m_headers + "\r\n" );
}


void FalhttpdReply::send( const Falcon::String& s )
{
   Falcon::uint32 sent = 0;
   while( sent < s.size() )
   {
      int res = ::send( m_socket, (const char*) s.getRawStorage() + sent, s.size() - sent, 0 );
      if( res < 0 )
         return;
      sent += res;
   }
}


Falcon::CoreObject* FalhttpdReply::factory( const Falcon::CoreClass* cls, void* ud, bool bDeser )
{
   return new FalhttpdReply( cls );
}



/* falhttpd_reply.cpp */
