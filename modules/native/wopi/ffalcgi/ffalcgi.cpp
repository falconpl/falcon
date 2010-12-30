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


#include <fastcgi.h>
//We're not an application program; instead, we're using the FCGI api directly
#define NO_FCGI_DEFINES
#include <fcgi_stdio.h>

#include <stdio.h>

#include <falcon/engine.h>
#include <falcon/stdstreams.h>
#include <falcon/sys.h>
#include <falcon/wopi/wopi_ext.h>

#include <cgi_options.h>
#include <cgi_request.h>
#include <cgi_reply.h>
#include <falcgi_perform.h>

static void report_temp_file_error( const Falcon::String& fileName, void* data )
{
   Falcon::AutoCString cstr(fileName);
   printf( "ERROR: Cannot remove temp file %s\r\n", cstr.c_str() );
}

int main( int argc, char* argv[] )
{
   // we need to re-randomize based on our pid + time,
   // or we may end up creating the same temp files.
   Falcon::WOPI::Utils::xrandomize();

   // start the engine
   Falcon::Engine::Init();

   CGIOptions cgiopt;
   if ( cgiopt.init( argc, argv ) )
   {
      while( FCGI_Accept() >= 0 )
      {
         void *tempFileList = perform( cgiopt, argc, argv );

         // perform complete GC to reset open states (i.e. open file handles).
         Falcon::memPool->performGC();

         // Free the temp files
         if( tempFileList != 0 )
         {
            Falcon::WOPI::Request::removeTempFiles( tempFileList, 0, report_temp_file_error );
         }
      }
   }

   delete cgiopt.m_smgr;

   // engine down.
   Falcon::Engine::Shutdown();

   return 0;
}
