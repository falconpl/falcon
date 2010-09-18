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
#include "falhttpd_reply.h"
#include <falcon/engine.h>
#include <falcon/wopi/wopi_ext.h>

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

FalhttpdApp* FalhttpdApp::m_theApp = 0;

FalhttpdApp::FalhttpdApp():
   m_logModule(0),
   m_log(0)
{
   if ( m_theApp != 0 )
   {
      throw std::runtime_error( "FalhttpdApp is not singleton");
   }

   m_theApp = this;

   // start the engine
   Falcon::Engine::Init();
   Falcon::Engine::cacheModules( true );

   // let's create also a useful general module loader.
   m_loader = new Falcon::ModuleLoader(".");
   m_loader->addFalconPath();

   Falcon::String io_encoding;
   m_ss = Falcon::stdOutputStream();

   m_coreModule = Falcon::core_module_init();
   m_wopiModule = Falcon::WOPI::wopi_module_init(
         &WOPI::CoreRequest::factory,
         &FalhttpdReply::factory );
}


FalhttpdApp::~FalhttpdApp()
{

   if ( m_log != 0 )
   {
      // we did get initialized
      m_log->log(LOGLEVEL_INFO, "Log closed." );
      m_log->decref();
   }

   if( m_logModule != 0 )
   {
      m_logModule->decref();
   }

   m_wopiModule->decref();
   m_coreModule->decref();

   delete m_ss;

   if( m_nSocket != 0 )
   {
      #ifdef _WIN32
         closesocket( m_nSocket );
         WSACleanup();
      #else
         ::close( m_nSocket );
      #endif
   }

   Falcon::Engine::Shutdown();
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

   if( ! readyServices() )
      return false;

   if( m_hopts.m_loadPath.size() )
   {
      m_loader->setSearchPath( m_hopts.m_loadPath );
   }

   return readyNet();
}


bool FalhttpdApp::readyServices()
{
   // we'll be throwing an error if not available.
   m_logModule = m_loader->loadName("logging");
   Falcon::LogService* ls = (Falcon::LogService*) m_logModule->getService(LOGSERVICE_NAME);
   fassert( ls != 0 );

   // create our application log area.
   m_log = ls->makeLogArea( "HTTP" );
   readyLog( ls );
   m_log->log( LOGLEVEL_INFO, "Falcon HTTPD server started..." );

   if( m_hopts.m_configFile.size() != 0 )
   {
      // Ok, time to get the config (we don't need to cache it)
      Falcon::Module* cfgmod = m_loader->loadName( "confparser" );
      Falcon::ConfigFileService* cfs = (Falcon::ConfigFileService*) cfgmod->getService( CONFIGFILESERVICE_NAME );
      fassert( cfs != 0 );

      if( ! cfs->initialize( m_hopts.m_configFile, "utf-8" ) || ! cfs->load() )
      {
         m_log->log( LOGLEVEL_WARN, "Cannot read init file "
               + m_hopts.m_configFile + ": " + cfs->errorMessage() );
      }
      else
      {
         m_hopts.loadIni( m_log, cfs );
      }
      // Not working because of static destructor. We need to upgrate core.
      cfs->clearMainSection();
      cfgmod->decref();
   }

   return true;
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


void FalhttpdApp::readyLog( Falcon::LogService* ls )
{
   // and a console channel logging everything up to info level
   if( ! m_hopts.m_bQuiet )
   {
      Falcon::LogChannel* chnConsole = ls->makeChnStream(
            Falcon::stdOutputStream(),
            "[%S %L|%a] %m",
            m_hopts.m_logLevel );

      m_log->addChannel( chnConsole );
      chnConsole->decref();
   }

   if( m_hopts.m_bSysLog )
   {
      Falcon::LogChannel* chnSyslog = ls->makeChnSyslog(
               "FHTTPD",
               "[%S %L|%a] %m",
               0,
               m_hopts.m_logLevel );
      m_log->addChannel( chnSyslog );
      chnSyslog->decref();
   }

   if( m_hopts.m_sLogFiles.size() != 0 )
   {
      // on files, log everything
      Falcon::LogChannel* chnStream = ls->makeChnlFiles( m_hopts.m_sLogFiles );
      m_log->addChannel( chnStream );
      chnStream->decref();
   }
}

void FalhttpdApp::usage()
{
   m_ss->writeString( "  Usage: \n"
      "       falhttpd [options]\n\n"
      "  Options:\n"
      "  -?          This help\n"
      "  -A <dir>    Directory where to place persistent application data.\n"
      "  -C <file>   Load this configuration file\n"
      "  -D <file>   Log debug level informations to the given file or path\n"
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
      SOCKET sIncoming = ::accept( m_nSocket, (struct sockaddr*) &inAddr, &inLen );

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

      FalhttpdClient* cli = new FalhttpdClient( m_hopts, m_log, sIncoming, sRemote );
      cli->serve();
      delete cli;

   }

   return 0;
}


//=============================================================================
//

int main( int argc, char* argv[] )
{
   FalhttpdApp theApp;

   try
   {
      if( theApp.init( argc, argv ) )
      {
         return theApp.run();
      }
      else
      {
         theApp.m_ss->writeString(
               Falcon::String( "falhttpd: Cannot intialize application.\n") +
               Falcon::String( "falhttpd: ") + theApp.m_hopts.m_sErrorDesc +"\n\n" );
      }
   }
   catch( Falcon::Error* e )
   {
      // Run catches its own error. Errors here can happen only during initialization
      theApp.m_ss->writeString( "Error during initialization: \n" );
      theApp.m_ss->writeString( e->toString() + "\n" );
      theApp.m_ss->flush();
      e->decref();
   }

   return -1;
}

/* end of falhttpd.cpp */

