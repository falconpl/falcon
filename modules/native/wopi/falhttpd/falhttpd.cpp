/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 23 Feb 2010 22:09:02 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "falhttpd.h"
#include "falhttpd_client.h"
#include <falcon/engine.h>
#include <falcon/autocstring.h>
#include <inet_mod.h>
#include <inet_ext.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifndef FALCON_SYSTEM_WIN
#include <unistd.h>
#include <arpa/inet.h>      /* inet_ntoa() to format IP address */
#include <netinet/in.h>     /* in_addr structure */
#include <netdb.h>
#else
#include <ws2tcpip.h>
#define socklen_t int
#endif


namespace Falcon {

FalhttpdApp* FalhttpdApp::m_theApp = 0;

FalhttpdApp::FalhttpdApp():
   m_log(0),
   m_ll(0)
{
   if ( m_theApp != 0 )
   {
      throw std::runtime_error( "FalhttpdApp is not singleton");
   }

   m_theApp = this;
   m_log = Engine::instance()->log();
   m_ss = new TextWriter( new StdOutStream );
   Mod::SocketStream::setHandler( new Ext::ClassSocketStream );
}


FalhttpdApp::~FalhttpdApp()
{
   if( m_nSocket != 0 )
   {
      #ifdef _WIN32
         closesocket( m_nSocket );
         WSACleanup();
      #else
         ::close( m_nSocket );
      #endif
   }

   m_ss->decref();
}



bool FalhttpdApp::init( int argc, char* argv[] )
{
   // configure command line
   if( ! m_hopts.init( argc, argv ) )
   {
      usage();
      return false;
   }

   if( m_hopts.m_bHelp )
   {
      usage();
   }

   return   readyLog()
         && readyNet()
         && readyVM()
            ;
}


bool FalhttpdApp::readyNet()
{
   struct sockaddr_in sa;

#ifdef _WIN32
   WORD wVersionRequested = MAKEWORD(2, 1);
   WSADATA WSAData;

   if( WSAStartup( wVersionRequested, &WSAData ) != 0 )
   {
      m_hopts.m_sErrorDesc = "Cannot initialize Winsock 2.1";
      return false;
   }
#endif


   ::memset(&sa, 0, sizeof(struct sockaddr_in) );

   Falcon::AutoCString myname( m_hopts.m_sIface );
   sa.sin_family = AF_INET;
   sa.sin_addr.s_addr = inet_addr( myname.c_str() );
   sa.sin_port = htons( m_hopts.m_nPort );
   if (( m_nSocket = ::socket( sa.sin_family, SOCK_STREAM, 0) ) < 0)
   {
      m_hopts.m_sErrorDesc = "Cannot create the listening socket (";
      m_hopts.m_sErrorDesc.N( (Falcon::int64) m_nSocket ).A(")");
      m_nSocket = 0;
      return false;
   }

   int val = 1;
   ::setsockopt(m_nSocket, SOL_SOCKET, SO_REUSEADDR, (const char*) &val, sizeof(val));

   if ( ::bind( m_nSocket, (struct sockaddr*) &sa, sizeof(struct sockaddr_in) ) < 0)
   {
      m_hopts.m_sErrorDesc = "Can't bind to local address " + m_hopts.m_sIface + ":";
      m_hopts.m_sErrorDesc.N( m_hopts.m_nPort );
      return false;
   }

   ::listen( m_nSocket, 10 );

   return true;
}


bool FalhttpdApp::readyLog()
{
   // and a console channel logging everything up to info level
   if( ! m_hopts.m_bQuiet )
   {
      Stream* stream = new StdErrStream;
      m_ll = new LogListener( stream );
      stream->decref();
      m_log->addListener(m_ll);
   }

   if( m_hopts.m_bSysLog )
   {
      // todo
   }

   if( m_hopts.m_sLogFiles.size() != 0 )
   {
      // on files, log everything
      // todo
   }
   return true;
}

bool FalhttpdApp::readyVM()
{
   return true;
}

void FalhttpdApp::usage()
{
   LocalRef<Stream> out( new StdOutStream );
   TextWriter tw(out);

   tw.write( "  Usage: \n"
      "       falhttpd [options]\n\n"
      "  Options:\n"
      "  -?          This help\n"
      "  -A <dir>    Directory where to place persistent application data.\n"
      "  -C <file>   Load this configuration file\n"
      "  -D <file>   Log debug level informations to the given file or path\n"
      "  -e <enc>    Text encoding used to render text output from scripts.\n"
      "  -E <enc>    Text encoding used to read source scripts (defaults to -e).\n"
      "  -h <dir>    Sets this directory as site HTDOCS root\n"
      "  -i <addr>   Listen on the named interface (defaults to 0.0.0.0 - all)\n"
      "  -l <n>      Log level (0 min, 5 max)\n"
      "  -L <path>   Set this as the falcon load path\n"
      "  -p <port>   Listen on required port (deaults to 80)\n"
      "  -q          Be quiet (don't log on console)\n"
      "  -S          Do not log on syslog\n"
      "  -t <secs>   Set session timeout (defaults to 30)\n"
      "  -T <dir>    Use this as temporary path\n\n"
   );
}

int FalhttpdApp::run()
{
   // accept.
   while( true )
   {
      sockaddr_in inAddr;
      socklen_t inLen = sizeof(inAddr);
      SOCKET sIncoming = ::accept( m_nSocket, (struct sockaddr*) &inAddr, (unsigned int*)&inLen );

      logi( "Incoming client" );
      if( sIncoming == 0 )
      {
         // we're done
         break;
      }

      char host[256];
      char serv[32];
      Falcon::String sRemote;

      int error = ::getnameinfo( (struct sockaddr*) &inAddr, inLen, host, 255, serv, 31, NI_NUMERICHOST | NI_NUMERICSERV );

      if( error == 0 )
      {
         sRemote = host;
         sRemote += ":";
         sRemote += serv;
      }
      else
      {
         sRemote = "unknown";
      }

      Mod::Socket* skt = new Mod::Socket( sIncoming, PF_INET, SOCK_STREAM, 0, new Mod::Address(sRemote) );

      FalhttpdClient* cli = new FalhttpdClient( m_hopts, skt );
      cli->serve();
      delete cli;

   }

   return 0;
}


String FalhttpdApp::codeDesc( int errorID )
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


String FalhttpdApp::serverSignature()
{
   //TODO Make this configurable
   String sString = "<br/><hr/>\n<p><i>Falcon HTTPD simple server.</i></p>\n";
   return sString;
}

}

//=============================================================================
//
int main( int argc, char* argv[] )
{
   Falcon::Engine::init();
   Falcon::FalhttpdApp theApp;

   try
   {
      if( theApp.init( argc, argv ) )
      {
         return theApp.run();
      }
      else
      {
         theApp.m_ss->writeLine(
               Falcon::String( "falhttpd: Cannot intialize application.\n") +
               Falcon::String( "falhttpd: ") + theApp.m_hopts.m_sErrorDesc +"\n" );
      }
   }
   catch( Falcon::Error* e )
   {
      // Run catches its own error. Errors here can happen only during initialization
      theApp.m_ss->write( "Error during initialization: \n" );
      theApp.m_ss->write( e->describe() + "\n" );
      theApp.m_ss->flush();
      e->decref();
   }

   Falcon::Engine::shutdown();
   return -1;
}



/* end of falhttpd.cpp */

