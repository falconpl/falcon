/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_client.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Servicing single client.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 24 Feb 2010 20:10:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define MAX_HEADER_SIZE 100000

#include <inet_mod.h>
#include <falcon/autocstring.h>

#include "falhttpd.h"
#include "falhttpd_client.h"
#include "falhttpd_rh.h"

#ifndef _WIN32
#include <unistd.h>
#include <arpa/inet.h>      /* inet_ntoa() to format IP address */
#include <netinet/in.h>     /* in_addr structure */
#include <netdb.h>
#else
// #pragma comment(lib, "wininet.lib")
#endif

#include <falcon/wopi/request.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/stream_ch.h>
#include <falcon/sys.h>

namespace Falcon {

FalhttpdClient::FalhttpdClient( const FalhttpOptions& options, Mod::Socket* skt ):
   Client( new WOPI::Request, new WOPI::Reply, new StreamBuffer(skt->makeStreamInterface()) ),
   m_skt( skt ),
   m_options( options )
{
   LOGI( "Incoming client from "+ skt->address()->toString() );
   m_reply->setCommitHandler( new WOPI::StreamCommitHandler(m_stream) );
}


FalhttpdClient::~FalhttpdClient()
{
}


void FalhttpdClient::close()
{
   if( m_skt != 0 )
   {
      m_stream->flush();
      LOGI( "Client "+ m_skt->address()->toString() + " gone" );
      m_stream->decref();
      m_skt->close();
      m_skt->decref();
      m_skt = 0;
   }
}

void FalhttpdClient::serve()
{
   LOGD( "Serving client " + m_skt->address()->toString() );

   // get the header
   String sHeader;
   StreamBuffer& si = *static_cast<StreamBuffer*>(m_stream);
   uint32 chr;
   while ( ! sHeader.endsWith("\r\n") && si.get(chr) )
   {
      // do nothing
      sHeader.append( chr );
   }

   if ( ! sHeader.endsWith("\r\n") )
   {
      FalhttpdApp::get()->logd( "Client "+ m_skt->address()->toString() + " has sent an invalid header." );
      return;
   }
   // remove trailing \r\n
   sHeader.remove( sHeader.length()-2, 2 );

   // parse the header; must be in format GET/POST/HEAD 'uri' HTTP/1.x
   uint32 p1, p2;
   p1 = sHeader.find( " " );
   p2 = sHeader.rfind( " " );
   if ( p1 == p2 )
   {
      sHeader.trim();
      LOGW( "Client "+ m_skt->address()->toString() + " has sent an invalid header: " + sHeader );
      return;
   }

   String sMethod = sHeader.subString( 0, p1 );
   String sUri = sHeader.subString( p1+1, p2 );
   String sProto = sHeader.subString( p2+1 );

   // a bit of formal control
   int type = sMethod == "GET" ? 1 :
               ( sMethod == "POST"  ? 2 : ( sMethod == "HEAD" ? 3 : 0 ) );

   if ( type == 0 )
   {
      LOGW( "Client "+ m_skt->address()->toString() + " has sent an invalid type: " + sHeader );
      FalhttpdApp::get()->eh()->renderSysError( this, 400, "Invalid requests type" );
      return;
   }

   URI uri( sUri );
   if( ! uri.isValid() )
   {
      LOGW( "Client "+ m_skt->address()->toString() + " has sent an invalid URI: " + sHeader );
      FalhttpdApp::get()->eh()->renderSysError( this, 400, "Invalid invalid uri" );
      return;
   }

   if( sProto != "HTTP/1.0" && sProto != "HTTP/1.1" )
   {
      LOGW( "Client "+ m_skt->address()->toString() + " has sent an invalid Protocol: " + sHeader );
      FalhttpdApp::get()->eh()->renderSysError( this, 505, "Invalid protocol in header" );
      return;
   }

   // ok, we got a valid header -- proceed in serving the request.
   serveRequest( sMethod, sUri, sProto );
}


void FalhttpdClient::serveRequest(
      const String& sMethod, const String& sUri,  const String& sProto )
{
   LOGI( "Serving request from "+ m_skt->address()->toString() + ": " + sMethod + " " + sUri + " " + sProto );

   // first, read the headers.

   WOPI::Request* req = m_request;
   req->startedAt( Sys::_seconds() );

   req->setURI( sUri );
   req->m_remote_ip = m_skt->address()->toString();
   String sFile = req->parsedUri().path().encode();

   LOGD( "Remapping file "+ sFile );
   // Time to re-map the file
   if ( ! m_options.remap( sFile ) )
   {
      LOGW( "Not found file "+ sFile );
      FalhttpdApp::get()->eh()->renderSysError( this, 404, "" );
   }
   else
   {
      LOGD( "File remapped as "+ sFile );
      req->m_filename = sFile;

      // and finally process the request through the appropriate request handler.
      FalhttpdRequestHandler* rh = m_options.getHandler( sFile, this );
      rh->serve();
      delete rh;
   }
   LOGI( "Served client "+ m_skt->address()->toString() );
}


void FalhttpdClient::replyError( int code, const String& msg )
{
   FalhttpdApp::get()->eh()->renderSysError(this, code, msg);
}

}

/* falhttpd_client.cpp */
