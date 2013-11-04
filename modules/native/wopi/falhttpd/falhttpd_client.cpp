/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_client.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Servicing single client.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 24 Feb 2010 20:10:45 +0100
s
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
   m_skt( skt ),
   m_bComplete( false ),
   m_options( options ),
   m_bDeleteReply(true)
{
   LOGI( "Incoming client from "+ skt->address()->toString() );
   m_stream = new StreamBuffer(m_skt->makeStreamInterface());
   m_reply = new WOPI::Reply;
   m_reply->setCommitHandler( new WOPI::StreamCommitHandler(m_stream) );
}

FalhttpdClient::~FalhttpdClient()
{
   close();
   if( m_bDeleteReply )
      delete m_reply;
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
   StreamBuffer& si = *m_stream;
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
      replyError( 400, "Invalid requests type" );
      return;
   }

   URI uri( sUri );
   if( ! uri.isValid() )
   {
      LOGW( "Client "+ m_skt->address()->toString() + " has sent an invalid URI: " + sHeader );
      replyError( 400, "Invalid invalid uri" );
      return;
   }

   if( sProto != "HTTP/1.0" && sProto != "HTTP/1.1" )
   {
      LOGW( "Client "+ m_skt->address()->toString() + " has sent an invalid Protocol: " + sHeader );
      replyError( 505 );
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

   WOPI::Request* req = new WOPI::Request;
   req->startedAt( Sys::_seconds() );

   req->setURI( sUri );
   req->m_remote_ip = m_skt->address()->toString();
   String sFile = req->parsedUri().path().encode();

   LOGD( "Remapping file "+ sFile );
   // Time to re-map the file
   if ( ! m_options.remap( sFile ) )
   {
      LOGW( "Not found file "+ sFile );
      replyError( 404 );
      delete req;
   }
   else
   {
      LOGD( "File remapped as "+ sFile );
      req->m_filename = sFile;

      // and finally process the request through the appropriate request handler.
      FalhttpdRequestHandler* rh = m_options.getHandler( sFile, this );
      rh->serve( req );
      delete rh;
   }
   LOGI( "Served client "+ m_skt->address()->toString() );
}


void FalhttpdClient::replyError( int errorID, const String& explain )
{
   String sErrorDesc = codeDesc( errorID );
   String sReply;
   String sError;
   m_reply->reason(sErrorDesc);
   m_reply->status(errorID);

   // for now, we create the docuemnt here.
   // TODO Read the error document from a file.
   String sErrorDoc = "<html>\n"
         "<head>\r\n"
         "<title>Error " +sError + "</title>\n"
         "</head>\n"
         "<body>\n"
         "<h1>" + sError + "</h1>\n";

   if( explain.size() != 0 )
   {
      sErrorDoc += "<p>This abnormal condition has been encountered while receiving and processing Your request:</p>\n";
      sErrorDoc += "<p><b>" + explain + "</b></p>\n";
   }
   else
   {
      sErrorDoc += "<p>An error of type <b>" + sError + "</b> has been detected while parsing your request.</p>\n";
   }

   sErrorDoc += getServerSignature();
   sErrorDoc += "</body>\n</html>\n";

   AutoCString content( sErrorDoc );

   TimeStamp now;
   now.currentTime();
   m_reply->setHeader( "Date", now.toRFC2822() );

   m_reply->setHeader( "Content-Length", String().N( (int64) content.length() ));
   m_reply->setContentType("text/html; charset=utf-8");

   LOGI( "Sending ERROR reply to client " + m_skt->address()->toString() + ": " + sError );
   sendData( content.c_str(), content.length() );

}

String FalhttpdClient::getServerSignature()
{
   //TODO Make this configurable
   String sString = "<br/><hr/>\n<p><i>Falcon HTTPD simple server.</i></p>\n";

   return sString;
}


void FalhttpdClient::consumeRequest()
{
   unsigned char buffer[2048];
   while( m_skt->recv(buffer,2048) == 2048);
}


String FalhttpdClient::codeDesc( int errorID )
{
   switch( errorID )
   {
   // continue codes
   case 100: return "Continue";
   case 101: return "Switching Protocols";

   // Success codes
   case 200: return "OK";
   case 201: return "Created";
   case 202: return "Accepted";
   case 203: return "Non-Authoritative Information";
   case 204: return "No Content";
   case 205: return "Reset Content";
   case 206: return "Partial Content";

   // Redirection Codes
   case 300: return "Multiple Choices";
   case 301: return "Moved Permanently";
   case 302: return "Found";
   case 303: return "See Other";
   case 304: return "Not Modified";
   case 305: return "Use Proxy";
   case 307: return "Temporary Redirect";

   // Client Error Codes

   case 400: return "Bad Request";
   case 401: return "Unauthorized";
   case 402: return "Payment Required";
   case 403: return "Forbidden";
   case 404: return "Not Found";
   case 405: return "Method Not Allowed";
   case 406: return "Not Acceptable";
   case 407: return "Proxy Authentication Required";
   case 408: return "Request Timeout";
   case 409: return "Conflict";
   case 410: return "Gone";
   case 411: return "Length Required";
   case 412: return "Precondition Failed";
   case 413: return "Request Entity Too Large";
   case 414: return "Request-URI Too Large";
   case 415: return "Unsupported Media Type";
   case 416: return "Requested Range Not Satisfiable";
   case 417: return "Expectation Failed";

   // Server Error Codes
   case 500: return "Internal Server Error";
   case 501: return "Not Implemented";
   case 502: return "Bad Gateway";
   case 503: return "Service Unavailable";
   case 504: return "Gateway Timeout";
   case 505: return "HTTP Version not supported";

   }

   return "Unknown code";
}


void FalhttpdClient::sendData( const String& sReply )
{
   sendData( sReply.getRawStorage(), sReply.size() );
}


void FalhttpdClient::sendData( const void* data, uint32 size )
{
   uint32 sent = 0;
   int res = 0;
   const byte* bdata = (const byte* ) data;

   m_reply->commit();

   while( sent < size )
   {
      res = m_stream->write( bdata + sent, size - sent );
      if( res < 0 )
      {
         LOGW( "Client "+ m_skt->address()->toString() + " had a send error." );
         return;
      }
      sent += res;
   }
}

}

/* falhttpd_client.cpp */
