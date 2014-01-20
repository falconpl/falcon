/*
   FALCON - The Falcon Programming Language.
   FILE: scriptrunner.cpp

   Web Oriented Programming Interface.

   Utility to configure and run a WOPI-oriented script process.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 15 Jan 2014 12:36:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>
#include <falcon/sys.h>
#include <falcon/wopi/modulewopi.h>
#include <falcon/wopi/replystream.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/classrequest.h>
#include <falcon/wopi/stream_ch.h>
#include <falcon/wopi/errorhandler.h>
#include <falcon/wopi/client.h>

#include <falcon/wopi/scriptrunner.h>

namespace Falcon {
namespace WOPI {

#define LOGF(__x__) do{ m_log->log(Log::fac_app, Log::lvl_critical, __x__); } while(0)
#define LOGE(__x__) do{ m_log->log(Log::fac_app, Log::lvl_error, __x__); } while(0)
#define LOGW(__x__) do{ m_log->log(Log::fac_app, Log::lvl_warn, __x__); } while(0)
#define LOGI(__x__) do{ m_log->log(Log::fac_app, Log::lvl_info, __x__); } while(0)
#define LOGD(__x__) do{ m_log->log(Log::fac_app, Log::lvl_detail, __x__); } while(0)


ScriptRunner::ScriptRunner( const String& provider, VMachine* vm, Log* log, ErrorHandler* eh ):
   m_provider(provider),
   m_vm(vm),
   m_log(log),
   m_eh( eh )
{
}

ScriptRunner::~ScriptRunner()
{
}


void ScriptRunner::run( Client* client, const String& localScript, const Wopi* cfgTemplate )
{
   Process* process = m_vm->createProcess();

   // declare the process main context already active, so that the GC
   // won't mark/sweep till we put the context in execution (or kill it).
   process->mainContext()->setStatus(VMContext::statusReady);

   // we must have already checked for the engine having this encoding.
   process->setStdEncoding(m_sTextEncoding);

   // same applies here.
   ModLoader* ml = process->modSpace()->modLoader();
   ml->sourceEncoding( m_sSourceEncoding );


   if( m_loadPath.size() == 0 )
      ml->addFalconPath();
   else
      ml->addSearchPath( m_loadPath );

   Falcon::Path path( localScript );
   ml->addDirectoryFront( path.fulloc() );

   Module* core = new CoreModule;
   WOPI::Request* req = client->request();
   WOPI::Reply* reply = client->reply();

   WOPI::ModuleWopi* modwopi = new WOPI::ModuleWopi(m_provider, req, reply );
   client->detach();

   process->modSpace()->add( core );
   // we don't need an extra ref for core
   core->decref();
   process->modSpace()->add( modwopi );
   // we DO need an extra ref for wopi, as we'll use it after the process is gone.
   if( cfgTemplate != 0 ) {
      modwopi->wopi()->configFromWopi( *cfgTemplate );
   }

   WOPI::ReplyStream* r_stdout = new WOPI::ReplyStream(modwopi->reply(), client->stream(), false );
   WOPI::ReplyStream* r_stderr = new WOPI::ReplyStream(modwopi->reply(), process->stdErr(), false );
   process->stdOut(r_stdout);
   process->stdErr(r_stderr);
   r_stdout->decref();
   r_stderr->decref();

   modwopi->scriptName(path.filename());
   modwopi->scriptPath(localScript);

   // now that the module (and so, the request) is fully configured,
   // we can read the incoming data
   try
   {
      req->parse( client->stream() );
   }
   catch(Error* err)
   {
      if( req->m_content_length != req->m_bytes_received )
      {
         client->consumeRequest();
      }
      m_eh->renderSysError( client, 400, err->describe(false) );
      err->decref();
      process->decref();
      modwopi->decref();
      return;
   }


   try
   {
      modwopi->wopi()->setupLogListener();

      Process* loadProc = process->modSpace()->loadModule( localScript, true, true, true );
      LOGI( String("Starting loader process on: ") + localScript );
      process->modSpace()->resolveDeps(loadProc->mainContext(), modwopi );
      loadProc->start();
      loadProc->wait();
      LOGI( String("Completed loading script on: ") + localScript );

      // get the main module
      Module* mod = static_cast<Module*>(loadProc->result().asInst());
      loadProc->decref();

      String configErrors;
      if ( ! modwopi->wopi()->configFromModule( mod, configErrors ) )
      {
         String text = "Invalid configuration in module " + localScript + ":\n" + configErrors;
         LOGW( text );
         m_eh->renderSysError( client, 500, text );
         process->decref();
         modwopi->decref();
         return;
      }

      Function* mainFunc = mod->getMainFunction();
      if( mainFunc != 0 )
      {
         LOGI( String("Launching main script function on: ") + localScript );
         mod->addMantra( modwopi->wopi()->onTerminate, false );
         process->pushCleanup( modwopi->wopi()->onTerminate );
         process->start( mainFunc );
         process->wait();
         LOGI( "Script " + localScript + " completed." );
      }
      else {
         String text = "No main function to be processed in " + localScript;
         m_eh->renderSysError( client, 500, text );
      }
   }
   catch( Error* err )
   {
      m_eh->renderError( client, err );
      err->decref();
   }

   modwopi->reply()->commit();
   modwopi->wopi()->renderWebLogs(client->stream());
   modwopi->wopi()->removeLogListener();
   client->stream()->flush();
   process->decref();

   modwopi->wopi()->removeTempFiles();
   modwopi->decref();
}


}
}

/* end of scriptrunner.cpp */
