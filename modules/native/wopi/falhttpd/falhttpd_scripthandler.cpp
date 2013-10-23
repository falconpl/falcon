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
   WOPI::ModuleWopi* wopi = new WOPI::ModuleWopi("FalHTTPD", req);

   process->modSpace()->add( core );
   process->modSpace()->add( wopi );

   WOPI::ReplyStream* r_stdout = new WOPI::ReplyStream(wopi->reply(), process->stdOut() );
   WOPI::ReplyStream* r_stderr = new WOPI::ReplyStream(wopi->reply(), process->stdErr() );
   process->stdOut(r_stdout);
   process->stdErr(r_stderr);

   wopi->scriptName(path.filename());
   wopi->scriptPath(m_sFile);

   try
   {
      process->startScript( m_sFile );
      process->wait();
      LOGI( "Script " + m_sFile + " completed." );
   }
   catch( Error* err )
   {
      String s = err->describe();
      process->textErr()->write( s );
      LOGW( "Script "+ m_sFile + " terminated with error: " + s );
      err->decref();
   }

   wopi->reply()->commit( process->stdOut() );
   process->decref();
}

}

/* falhttpd_scripthandler.cpp */
