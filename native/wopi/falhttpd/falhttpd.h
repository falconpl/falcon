/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd.h

   Micro HTTPD server providing Falcon scripts on the web.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 23 Feb 2010 22:09:02 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_HTTPD_SERVER_H_
#define FALCON_HTTPD_SERVER_H_

#ifdef _WIN32
#define _WIN32_WINNT 0x0403
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/socket.h>
#define SOCKET int
#endif

#include <falcon/engine.h>
#include "falhttpd_options.h"

#include <falcon/srv/logging_srv.h>
#include <falcon/srv/confparser_srv.h>



class FalhttpdApp
{
public:
   FalhttpdApp();
   ~FalhttpdApp();

   bool init( int argc, char* argv[] );
   bool readyServices();
   bool readyNet();

   void usage();
   int run();

   inline void logf( const Falcon::String& l ) { m_log->log( LOGLEVEL_FATAL, l ); }
   inline void loge( const Falcon::String& l ) { m_log->log( LOGLEVEL_ERROR, l ); }
   inline void logw( const Falcon::String& l ) { m_log->log( LOGLEVEL_WARN, l ); }
   inline void logi( const Falcon::String& l ) { m_log->log( LOGLEVEL_INFO, l ); }
   inline void logd( const Falcon::String& l ) { m_log->log( LOGLEVEL_DEBUG, l ); }

   Falcon::Module* core() const { return m_coreModule; }
   Falcon::Module* wopi() const { return m_wopiModule; }
   Falcon::ModuleLoader* loader() const { return m_loader; }

   //! Sevice stream
   Falcon::Stream* m_ss;
   FalhttpOptions m_hopts;

   static FalhttpdApp* get() { return m_theApp; }

private:
   void readyLog( Falcon::LogService* );

   Falcon::Module* m_logModule;
   Falcon::LogArea* m_log;
   Falcon::ModuleLoader* m_loader;

   Falcon::Module* m_coreModule;
   Falcon::Module* m_wopiModule;

   SOCKET m_nSocket;

   static FalhttpdApp* m_theApp;
};


#endif

/* end of falhttpd.h */

