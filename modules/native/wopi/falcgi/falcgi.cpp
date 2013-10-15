/*
   FALCON - The Falcon Programming Language.
   FILE: falcgi.cpp

   Falcon CGI program driver - main file.

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


#include "cgi_options.h"
#include "cgi_request.h"
#include "falcgi_perform.h"

static void report_temp_file_error( const Falcon::String& fileName, void* data )
{
   Falcon::Stream* serr = (Falcon::Stream*) data;
   serr->write( "ERROR: Cannot remove temp file " + fileName +"\r\n");
   serr->flush();
}

int main( int argc, char* argv[] )
{
   // we need to re-randomize based on our pid + time,
   // or we may end up creating the same temp files.
   Falcon::WOPI::Utils::xrandomize();

   // start the engine
   Falcon::Engine::init();

   CGIOptions cgiopt;
   if ( cgiopt.init( argc, argv ) )
   {
      void *tempFileList = perform( cgiopt, argc, argv );

      // perform complete GC to reset open states (i.e. open file handles).
      Falcon::memPool->performGC();

      // Free the temp files
      if( tempFileList != 0 )
      {
         Falcon::Stream* errstream = new Falcon::TranscoderUTF8( Falcon::stdErrorStream() );
         Falcon::WOPI::Request::removeTempFiles( tempFileList, errstream, report_temp_file_error );
         delete errstream;
      }

   }

   // clear the manager
   delete cgiopt.m_smgr;

   // engine down.
   Falcon::Engine::shutdown();

   return 0;
}
