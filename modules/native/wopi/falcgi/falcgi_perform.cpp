/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_perform.cpp

   Falcon CGI program driver - part running a single script.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 17:35:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/engine.h>
#include <falcon/stdstreams.h>
#include <falcon/sys.h>
#include <falcon/wopi/wopi.h>
#include <falcon/wopi/file_sm.h>
#include <falcon/wopi/replystream.h>

#include <cgi_options.h>
#include <cgi_request.h>

#include <stdio.h>

void configure_loader( CGIOptions& opts, Falcon::ModuleLoader& ml )
{
   // TODO: use options
   Falcon::String lp;
   ml.addFalconPath();

   if( Falcon::Sys::_getEnv( "FALCON_LOAD_PATH", lp ) && lp != "" )
   {
      ml.addSearchPath( lp );
   }

   Falcon::Engine::setSearchPath( ml.getSearchPath() );

   ml.sourceEncoding("utf-8");
   Falcon::Engine::setEncodings( "utf-8", "utf-8" );
}

void* perform( CGIOptions& options, int argc, char* argv[] )
{
   double nStartedAt = Falcon::Sys::_seconds();

   // setup the virtual machine
   Falcon::VMachineWrapper vm;

   Falcon::ModuleLoader ml(".");
   configure_loader( options, ml );

   Falcon::Stream* input = makeInputStream();

   // ok, we're in the business.
   // Prepare the class
   vm->link( Falcon::core_module_init() );
   vm->link( Falcon::WOPI::wopi_module_init(CGIRequest::factory, CGIReply::factory ) );

   Falcon::Item* i_req = vm->findGlobalItem( "Request" );
   fassert( i_req != 0 );
   Falcon::Item* i_upld = vm->findGlobalItem( "Uploaded" );
   fassert( i_upld != 0 );
   Falcon::Item* i_reply = vm->findGlobalItem( "Reply" );
   fassert( i_reply != 0 );
   Falcon::Item* i_wopi = vm->findGlobalItem( "Wopi" );
   fassert( i_wopi != 0 );

   CGIRequest* request = Falcon::dyncast<CGIRequest*>( i_req->asObject() );
   Falcon::WOPI::Request *base = 0;
   Falcon::uint32 nSessionToken = 0;

   // And the reply here
   CGIReply* reply =  Falcon::dyncast<CGIReply*>( i_reply->asObject() );
   reply->init();

   vm->stdOut( new Falcon::WOPI::ReplyStream( reply ) );
   vm->stdErr( new Falcon::WOPI::ReplyStream( reply ) );

   // change the input (however, it's useless) -- but so the vm takes care of freeing it.
   vm->stdIn( input );
   vm->appSearchPath( ml.getSearchPath() );


   // From here, we may start to do wrong.
   try
   {
      Falcon::Module* main = ml.loadName(options.m_sMainScript);

      bool cfgSessionManager = false;;
      if( options.m_smgr == 0 )
      {
         cfgSessionManager = true;
         options.m_smgr = new Falcon::WOPI::FileSessionManager( "" );
      }

      // init must be called before the module can be configured.
      request->init( input, i_upld->asClass(), reply, options.m_smgr );
      // as it creates the base.
      base = request->base();

      base->startedAt( nStartedAt );
      base->setMaxMemUpload( options.m_maxMemUpload );
      if( options.m_sUploadPath != "" )
         base->setUploadPath( options.m_sUploadPath );
      nSessionToken = base->sessionToken();

      request->configFromModule( main );
      Falcon::dyncast<Falcon::WOPI::CoreWopi*>(i_wopi->asObject())->configFromModule( main );

      if( cfgSessionManager )
      {
         options.m_smgr->configFromModule( main );
         if( options.m_smgr->timeout() == 0 )
            options.m_smgr->timeout(600);
         options.m_smgr->startup();
      }

      Falcon::Path mainPath( options.m_sMainScript );
      // try to change, but ignore failure
      if ( mainPath.getFullLocation().size() != 0)
      {
         Falcon::int32 chdirError;
         Falcon::Sys::fal_chdir( mainPath.getFullLocation(), chdirError );
      }

      // having parsed the request, we can start.
      Falcon::Runtime rt( &ml );
      rt.addModule( main );
      vm->link( &rt );

      vm->launch();
   }
   catch( Falcon::Error* e )
   {
      Falcon::Stream* err = Falcon::stdErrorStream();
      // Write to the log ...
      err->writeString("FALCON ERROR:\r\n" +
            e->toString() );
      err->flush();

      reply->commit();
      //delete err;

      // ...and to the document
      vm->stdOut()->writeString("FALCON ERROR:\r\n" +
            e->toString()+"\r\n" );

      e->decref();
   }

   vm->stdOut()->close();
   vm->stdErr()->close();

   if( options.m_smgr != 0 && nSessionToken != 0 )
   {
      options.m_smgr->releaseSessions( nSessionToken );
   }

   return base != 0 ? base->getTempFiles() : 0;
   // here the VM is destroyed.
}

/* end of cgi_perform.cpp */

