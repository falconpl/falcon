/*
   FALCON - The Falcon Programming Language.
   FILE: client.cpp

   Web Oriented Programming Interface.

   Abstract representation of a client in the WOPI system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 15 Jan 2014 12:36:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#define MAX_HEADER_SIZE 100000

#include <falcon/autocstring.h>

#include <falcon/wopi/client.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/request.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/stream_ch.h>
#include <falcon/sys.h>

namespace Falcon {
namespace WOPI {

Client::Client( Request* request, Reply* reply, Stream* stream ):
   m_request(request),
   m_reply(reply),
   m_stream(stream),
   m_bComplete( false ),
   m_bOwnReqRep(true)
{}

Client::~Client()
{
   close();
   m_stream->decref();
   if( m_bOwnReqRep )
   {
      delete m_request;
      delete m_reply;
   }
}

void Client::close()
{
   m_stream->close();
}

void Client::consumeRequest()
{
   unsigned char buffer[2048];
   while( ! m_stream->eof() )
   {
      // Stream will throw an error in case of I/O error.
      m_stream->read(buffer,2048);
   }
}


void Client::sendData( const String& sReply )
{
   sendData( sReply.getRawStorage(), sReply.size() );
}


void Client::sendData( const void* data, uint32 size )
{
   uint32 sent = 0;
   int res = 0;
   const byte* bdata = (const byte* ) data;

   m_reply->commit();

   while( sent < size )
   {
      // Stream will throw an error in case of I/O error.
      res = m_stream->write( bdata + sent, size - sent );
      sent += res;
   }
}

}
}

/* client.cpp */
