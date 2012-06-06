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

#include "falhttpd.h"
#include "falhttpd_client.h"
#include "falhttpd_istream.h"
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
#include <falcon/wopi/mem_sm.h>
#include <falcon/sys.h>

FalhttpdClient::FalhttpdClient( const FalhttpOptions& options, LogArea* l, SOCKET s, const String& remName ):
   m_log( l ),
   m_nSocket( s ),
   m_sRemote( remName ),
   m_cBuffer( 0 ),
   m_sSize( 0 ),
   m_bComplete( false ),
   m_options( options )
{
   m_log->log( LOGLEVEL_INFO, "Incoming client from "+ m_sRemote );
   m_cBuffer = (char*) memAlloc( DEFAULT_BUFFER_SIZE );
   m_pSessionManager = options.m_pSessionManager;
}

FalhttpdClient::~FalhttpdClient()
{
   close();
}

void FalhttpdClient::close()
{
   if( m_nSocket != 0 )
   {
   #ifdef FALCON_SYSTEM_WIN
      ::shutdown( m_nSocket, SD_BOTH );
      ::closesocket( m_nSocket );
   #else
      ::shutdown( m_nSocket, SHUT_RDWR );
      ::close( m_nSocket );
   #endif

      m_log->log( LOGLEVEL_INFO, "Client "+ m_sRemote + " gone" );
      m_nSocket = 0;
   }
}

void FalhttpdClient::serve()
{
   m_log->log( LOGLEVEL_DEBUG, "Serving client "+ m_sRemote );

   // get the header
   String sHeader;
   StreamBuffer si( new SocketInputStream( m_nSocket ) );
   uint32 chr;
   while ( ! sHeader.endsWith("\r\n") && si.get(chr) )
   {
      // do nothing
      sHeader.append( chr );
   }

   if ( ! sHeader.endsWith("\r\n") )
   {
      m_log->log( LOGLEVEL_WARN, "Client "+ m_sRemote + " has sent an invalid header." );
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
      m_log->log( LOGLEVEL_WARN, "Client "+ m_sRemote + " has sent an invalid header: " + sHeader );
      return;
   }

   String sRequest = sHeader.subString( 0, p1 );
   String sUri = sHeader.subString( p1+1, p2 );
   String sProto = sHeader.subString( p2+1 );

   // a bit of formal control
   int type = sRequest == "GET" ? 1 :
               ( sRequest == "POST"  ? 2 : ( sRequest == "HEAD" ? 3 : 0 ) );

   if ( type == 0 )
   {
      m_log->log( LOGLEVEL_WARN, "Client "+ m_sRemote + " has sent an invalid type: " + sHeader );
      replyError( 400, "Invalid requests type" );
      return;
   }

   URI uri( sUri );
   if( ! uri.isValid() )
   {
      m_log->log( LOGLEVEL_WARN, "Client "+ m_sRemote + " has sent an invalid URI: " + sHeader );
      replyError( 400, "Invalid invalid uri" );
      return;
   }

   if( sProto != "HTTP/1.0" && sProto != "HTTP/1.1" )
   {
      m_log->log( LOGLEVEL_WARN, "Client "+ m_sRemote + " has sent an invalid Protocol: " + sHeader );
      replyError( 505 );
      return;
   }

   // ok, we got a valid header -- proceed in serving the request.
   serveRequest( sRequest, sUri, sProto, &si );
}

void FalhttpdClient::serveRequest(
      const String& sRequest, const String& sUri,  const String& sProto,
      Stream* si )
{
   m_log->log( LOGLEVEL_INFO, "Serving request from "+ m_sRemote + ": " + sRequest + " " + sUri + " " + sProto );

   // first, read the headers.

   WOPI::Request* req = new WOPI::Request;
   req->startedAt( Sys::_seconds() );
   if( ! req->parse( si ) )
   {
      replyError( 400, req->partHandler().error() );
      delete req;
      return;
   }
   
   m_log->log( LOGLEVEL_DEBUG, "Request parsed from "+ m_sRemote + " URI: " + sUri );
   req->setURI( sUri );
   req->m_remote_ip = m_sRemote;

   String sFile = req->parsedUri().path();

   m_log->log( LOGLEVEL_DEBUG, "Remapping file "+ sFile );
   // Time to re-map the file
   if ( ! m_options.remap( sFile ) )
   {
      m_log->log( LOGLEVEL_WARN, "Not found file "+ sFile );
      replyError( 404 );
   }
   else
   {
      m_log->log( LOGLEVEL_DEBUG, "File remapped as "+ sFile );
      req->m_filename = sFile;

      // and finally process the request through the appropriate request handler.
      FalhttpdRequestHandler* rh = m_options.getHandler( sFile, this );
      rh->serve( req );
      delete rh;
   }

   delete req;
   m_log->log( LOGLEVEL_INFO, "Served client "+ m_sRemote );
}


void FalhttpdClient::replyError( int errorID, const String& explain )
{
   String sErrorDesc = codeDesc( errorID );
   String sReply;
   String sError;
   sReply.A("HTTP/1.1 ").N( errorID ).A( " " + sErrorDesc + "\r\n");
   sError.N(errorID ).A( ": " + sErrorDesc );

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
   sReply += "Date: " + now.toRFC2822() + "\r\n";

   sReply.A( "Content-Length: ").N( (int64) content.length() ).A("\r\n");
   sReply += "Content-Type: text/html; charset=utf-8\r\n\r\n";

   m_log->log( LOGLEVEL_INFO, "Sending ERROR reply to client " + m_sRemote + ": " + sError );
   sendData( sReply );
   sendData( content.c_str(), content.length() );

}

String FalhttpdClient::getServerSignature()
{
   //TODO Make this configurable
   String sString = "<br/><hr/>\n<p><i>Falcon HTTPD simple server.</i></p>\n";

   return sString;
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
   const char* bdata = (const char* ) data;

   while( sent < size )
   {
      res = ::send( m_nSocket, bdata + sent, size - sent, 0 );
      if( res < 0 )
      {
         m_log->log( LOGLEVEL_WARN, "Client "+ m_sRemote + " had a send error." );
         return;
      }
      sent += res;
   }
}


/* falhttpd_socket.cpp */
