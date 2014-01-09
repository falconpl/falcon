/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_scripthandler.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Request handler processing a script.
   Mimetype is not determined (must be set by the script), but it
   defaults to text/html; charset=utf-8

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Mar 2010 12:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "falhttpd_scripthandler.h"
#include "falhttpd_client.h"
#include "falhttpd.h"

#include <falcon/falcon.h>
#include <falcon/sys.h>
#include <falcon/wopi/modulewopi.h>
#include <falcon/wopi/replystream.h>
#include <falcon/wopi/reply.h>
#include <falcon/wopi/classrequest.h>
#include <falcon/wopi/stream_ch.h>

namespace Falcon {

ScriptHandler::ScriptHandler( const Falcon::String& sFile, FalhttpdClient* cli ):
      FalhttpdRequestHandler( sFile, cli )
{
}


ScriptHandler::~ScriptHandler()
{

}


void ScriptHandler::serve( Falcon::WOPI::Request* req )
{
   FalhttpdApp* app = FalhttpdApp::get();
   Process* process = app->vm()->createProcess();

   // we must have already checked for the engine having this encoding.
   process->setStdEncoding(m_client->options().m_sTextEncoding);

   // same applies here.
   ModLoader* ml = process->modSpace()->modLoader();
   ml->sourceEncoding( m_client->options().m_sSourceEncoding );


   if( m_client->options().m_loadPath.size() == 0 )
      ml->addFalconPath();
   else
      ml->addSearchPath( m_client->options().m_loadPath );

   Falcon::Path path( m_sFile );
   ml->addDirectoryFront( path.fulloc() );

   Module* core = new CoreModule;
   WOPI::Reply* reply = m_client->reply();
   WOPI::ModuleWopi* wopi = new WOPI::ModuleWopi("FalHTTPD", req, reply );
   m_client->detachReply();

   process->modSpace()->add( core );
   process->modSpace()->add( wopi );
   wopi->wopi()->configFromWopi( m_client->options().m_templateWopi );

   WOPI::ReplyStream* r_stdout = new WOPI::ReplyStream(wopi->reply(), m_client->stream(), false );
   WOPI::ReplyStream* r_stderr = new WOPI::ReplyStream(wopi->reply(), process->stdErr(), false );
   wopi->reply()->setCommitHandler( new WOPI::StreamCommitHandler(m_client->stream()) );
   process->stdOut(r_stdout);
   process->stdErr(r_stderr);

   wopi->scriptName(path.filename());
   wopi->scriptPath(m_sFile);

   // now that the module (and so, the request) is fully configured,
   // we can read the incoming data


   // read the rest of the request.
   // Notice: the parsing disables the GC; our request object MUST be
   // already considered live in a VM.
   GCLock* lock = Engine::collector()->lock(Item(wopi->requestClass(), req));
   try
   {
      req->parse( m_client->stream() );
   }
   catch(Error* err)
   {
      if( req->m_content_length != req->m_bytes_received )
      {
         m_client->consumeRequest();
      }
      m_client->replyError( 400, err->describe(true) );
      err->decref();
      process->decref();
      return;
   }

   try
   {
      Process* loadProc = process->modSpace()->loadModule( m_sFile, true, true, true );
      LOGI( String("Starting loader process on: ") + m_sFile );
      process->modSpace()->resolveDeps(loadProc->mainContext(), wopi );
      loadProc->start();
      loadProc->wait();
      LOGI( String("Completed loading script on: ") + m_sFile );

      // get the main module
      Module* mod = static_cast<Module*>(loadProc->result().asInst());
      loadProc->decref();

      String configErrors;
      if ( ! wopi->wopi()->configFromModule( mod, configErrors ) )
      {
         String text = "Invalid configuration in module " + m_sFile + ":\n" + configErrors;
         LOGW( text );
         m_client->replyError( 500, text );
         process->decref();
         return;
      }

      Function* mainFunc = mod->getMainFunction();
      if( mainFunc != 0 )
      {
         LOGI( String("Launching main script function on: ") + m_sFile );
         process->start( mainFunc );
         process->wait();
         LOGI( "Script " + m_sFile + " completed." );
      }
      else {
         String text = "No main function to be processed in " + m_sFile;
         LOGW( text );
         m_client->replyError( 500, text );
         process->decref();
         return;
      }
   }
   catch( Error* err )
   {
      String s = err->describe();
      reply->setContentType("text/plain");
      process->textOut()->write( s );
      String text = "Script "+ m_sFile + " terminated with error: " + s;
      LOGW( text );
      err->decref();
   }

   lock->dispose();
   wopi->reply()->commit();
   process->decref();

   wopi->wopi()->removeTempFiles();
   wopi->decref();
}

}

/* falhttpd_scripthandler.cpp */
