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
#include "falhttpd_reply.h"
#include "falhttpd_ostream.h"
#include "falhttpd_client.h"
#include "falhttpd.h"

#include <falcon/wopi/wopi_ext.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/replystream.h>


ScriptHandler::ScriptHandler( const Falcon::String& sFile, FalhttpdClient* cli ):
      FalhttpdRequestHandler( sFile, cli )
{
}


ScriptHandler::~ScriptHandler()
{

}

static void report_temp_file_error( const Falcon::String& fileName, void* data )
{
   Falcon::LogArea* serr = (Falcon::LogArea*) data;
   serr->log( LOGLEVEL_WARN, "Cannot remove temp file " + fileName );
}

void ScriptHandler::serve( Falcon::WOPI::Request* req )
{
   Falcon::String cwd;
   Falcon::int32 status;
   Falcon::Sys::fal_getcwd( cwd, status );

   FalhttpdApp* app = FalhttpdApp::get();

   // find the file.
   Falcon::ModuleLoader ml( *app->loader() );
   ml.sourceEncoding( m_client->options().m_sSourceEncoding );
   Falcon::Engine::setEncodings( m_client->options().m_sSourceEncoding, m_client->options().m_sTextEncoding );

   if( m_client->options().m_loadPath.size() == 0 )
      ml.addFalconPath();
   else
      ml.addSearchPath( m_client->options().m_loadPath );

   Falcon::Path path( m_sFile );
   ml.addDirectoryFront( path.getFullLocation() );

   // broadcast the dir to other elements
   Falcon::Engine::setSearchPath( ml.getSearchPath() );

   void *tempFileList = 0;
   {
      Falcon::Runtime rt(&ml);
      Falcon::VMachineWrapper vm;

      // we should have no trouble here
      vm->link( app->core() );
      vm->link( app->wopi() );

      // Now let's init the request and reply objects
      Falcon::Item* i_req = vm->findGlobalItem( "Request" );
      Falcon::Item* i_rep = vm->findGlobalItem( "Reply" );
      Falcon::Item* i_cupt = vm->findGlobalItem( "Uploaded" );
      Falcon::Item* i_wopi = vm->findGlobalItem( "Wopi" );
      
      fassert( i_req->isObject() && i_rep->isObject() );
      fassert( i_wopi != 0 && i_wopi->isObject() );

      FalhttpdReply* rep = dyncast<FalhttpdReply*>(i_rep->asObject());
      rep->init( m_client->socket() );

      Falcon::WOPI::CoreRequest* creq = dyncast<Falcon::WOPI::CoreRequest*>(i_req->asObject());
      creq->init(i_cupt->asClass(), rep, m_client->smgr(), req );
      creq->processMultiPartBody();

      creq->provider("HTTPD");

      uint32 sTokenId = dyncast<Falcon::WOPI::CoreRequest*>(i_req->asObject())->base()->sessionToken();

      vm->stdOut( new Falcon::WOPI::ReplyStream( rep ) );
      vm->stdErr( new Falcon::WOPI::ReplyStream( rep ) );
      
      Falcon::Item* i_sn = vm->findGlobalItem( "scriptName" );
      Falcon::Item* i_sp = vm->findGlobalItem( "scriptPath" );
      *i_sn = new CoreString( path.getFullLocation() );
      *i_sp = new CoreString( path.getFilename() );

      if( m_client->options().m_sAppDataDir != "" )
         dyncast<Falcon::WOPI::CoreWopi*>( i_wopi->asObject() )->wopi()->dataLocation( m_client->options().m_sAppDataDir );

      try
      {
         // prepare the reply
         rt.loadFile( path.get() );
         vm->link( &rt );

         Falcon::Sys::fal_chdir( path.getFullLocation(), status );

         m_client->log()->log( LOGLEVEL_INFO, "Serving script "+ path.get() );

         vm->launch();
         // be sure that the output is committed, or output may be 0 below at close()
         rep->commit();
      }
      catch( Error* err )
      {
         rep->commit();

         String s = err->toString();
         rep->output()->writeString( s );
         rep->output()->flush();
         m_client->log()->log( LOGLEVEL_WARN, "Script "+ path.get()+ " terminated with error: " + s );
         err->decref();
      }

      tempFileList = req->getTempFiles();
      m_client->smgr()->releaseSessions(sTokenId);
      rep->output()->close();

   } // End of scope of the VM

   Falcon::Sys::fal_chdir( cwd, status );

   // Free the temp files
   if( tempFileList != 0 )
   {
      Falcon::WOPI::Request::removeTempFiles( tempFileList,
               m_client->log(),
               report_temp_file_error );
   }

}

/* falhttpd_scripthandler.cpp */
