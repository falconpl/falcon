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
#include <falcon/log.h>
#include <falcon/vm.h>
#include "falhttpd_options.h"
#include "falhttpd_eh.h"

namespace Falcon
{

   class TextWriter;
   class Stream;

   class FalhttpdApp
   {
   public:

      FalhttpdApp();
      ~FalhttpdApp();

      bool
      init(int argc, char* argv[]);

      void
      usage();
      int
      run();

      inline Log* log() const { return m_log; }

      inline void
      logf(const Falcon::String& l)
      {
         m_log->log(Log::fac_app, Log::lvl_critical, l);
      }
      inline void
      loge(const Falcon::String& l)
      {
         m_log->log(Log::fac_app, Log::lvl_error, l);
      }
      inline void
      logw(const Falcon::String& l)
      {
         m_log->log(Log::fac_app, Log::lvl_warn, l);
      }
      inline void
      logi(const Falcon::String& l)
      {
         m_log->log(Log::fac_app, Log::lvl_info, l);
      }
      inline void
      logd(const Falcon::String& l)
      {
         m_log->log(Log::fac_app, Log::lvl_detail, l);
      }

      Module*
      wopi() const
      {
         return m_wopiModule;
      }

      //! Sevice stream
      FalhttpOptions m_hopts;

      static FalhttpdApp*
      get()
      {
         return m_theApp;
      }

      TextWriter* m_ss;

      static String serverSignature();

      VMachine* vm() { return &m_vm; }

      SHErrorHandler* eh() const { return m_eh; }

   private:
      /**
       * The logger for the Falcon application.
       *
       * \TODO Add stream/stderr control in command line.
       */
      class LogListener: public Log::Listener
      {
      public:
         LogListener(Stream* stream);
         virtual
         ~LogListener();
         TextWriter* m_logfile;

      protected:

         virtual void
         onMessage(int fac, int lvl, const String& message);
      };

      Log* m_log;
      SHErrorHandler* m_eh;
      LogListener* m_ll;

      Falcon::Module* m_coreModule;
      Falcon::Module* m_wopiModule;

      SOCKET m_nSocket;
      VMachine m_vm;

      static FalhttpdApp* m_theApp;

      bool readyLog();
      bool readyEH();
      bool readyNet();
      bool readyVM();
   };

#define LOGF(__x__) do{ FalhttpdApp::get()->logf(__x__); } while(0)
#define LOGE(__x__) do{ FalhttpdApp::get()->loge(__x__); } while(0)
#define LOGW(__x__) do{ FalhttpdApp::get()->logw(__x__); } while(0)
#define LOGI(__x__) do{ FalhttpdApp::get()->logi(__x__); } while(0)
#define LOGD(__x__) do{ FalhttpdApp::get()->logd(__x__); } while(0)
}

#endif

/* end of falhttpd.h */

