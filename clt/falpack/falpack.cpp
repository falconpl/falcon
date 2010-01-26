/*
   FALCON - The Falcon Programming Language.
   FILE: falpack.cpp

   Packager for Falcon stand-alone applications
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Jan 2010 20:18:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>
#include "options.h"

using namespace Falcon;

String load_path;
Stream *stdIn;
Stream *stdOut;
Stream *stdErr;
String io_encoding;
bool ignore_defpath = false;


static void version()
{
   stdOut->writeString( "Falcon application packager.\n" );
   stdOut->writeString( "Version " );
   stdOut->writeString( FALCON_VERSION " (" FALCON_VERSION_NAME ")" );
   stdOut->writeString( "\n" );
   stdOut->flush();
}

static void usage()
{
   stdOut->writeString( "Usage: falpack [options] <main_script>\n" );
   stdOut->writeString( "\n" );
   stdOut->writeString( "Options:\n" );
   stdOut->writeString( "   -M          Pack also pre-compiled modules.\n" );
   stdOut->writeString( "   -s          Strip sources.\n" );
   stdOut->writeString( "   -e <enc>    Source files encoding.\n" );
   stdOut->writeString( "   -S          Do not store system files (engine + runner)\n" );
   stdOut->writeString( "   -r          Use falrun instead of falcon as runner\n" );
   stdOut->writeString( "   -P <dir>    Store non-application libraries in this subdirectory\n" );
   stdOut->writeString( "   -h, -?      This help.\n" );
   stdOut->writeString( "   -L <dir>    Redefine FALCON_LOAD_PATH\n" );
   stdOut->writeString( "   -v          Prints version and exit.\n" );
   stdOut->writeString( "\n" );
   stdOut->flush();
}


//TODO fill this per system
void transferSysFiles( Options &options )
{}


void transferModules( Options &options )
{
   ModuleLoader ml;

   ml.alwaysRecomp( true );
   ml.saveModules( false );

   if( options.m_sEncoding != "" )
      ml.sourceEncoding( options.m_sEncoding );

}


int main( int argc, char *argv[] )
{
   Falcon::GetSystemEncoding(io_encoding );

   if ( io_encoding != "" )
   {
      Transcoder *trans = TranscoderFactory( io_encoding, 0, true );
      if ( trans == 0 )
      {
         stdOut->writeString( "Fatal: unrecognized encoding '" + io_encoding + "'.\n\n" );
         return 1;
      }
      delete stdIn ;
      delete stdOut;
      delete stdErr;

      trans->setUnderlying( new StdInStream );

      stdIn = AddSystemEOL( trans, true );
      stdOut = AddSystemEOL( TranscoderFactory( io_encoding, new StdOutStream, true ), true );
      stdErr = AddSystemEOL( TranscoderFactory( io_encoding, new StdErrStream, true ), true );
   }

   Options options;

   if ( ! options.parse( argc-1, argv+1 ) )
   {
      stdOut->writeString( "Fatal: invalid parameters.\n\n" );
      return 1;
   }

   if( options.m_bVersion )
   {
      version();
   }

   if( options.m_bHelp )
   {
      usage();
   }

   // by default store the application in a subdirectory equal to the name of the
   // application.
   Path target( options.m_sMainScript );

   //===============================================================
   // We need a runtime and a module loader to load all the modules.
   transferModules( options );

   if( ! options.m_bNoSysFile )
      transferSysFiles( options );

   return 0;
}


/* end of falpack.cpp */

