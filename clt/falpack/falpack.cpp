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
#include <falcon/sys.h>
#include "options.h"

using namespace Falcon;

String load_path;
Stream *stdOut;
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
bool transferSysFiles( Options &options )
{
   return true;
}

bool storeModule( Options& options, Module* mod )
{
   //mod->path();

   // store it
   int fsStatus;
   if ( ! Sys::fal_mkdir( options.m_sTargetDir, fsStatus, true ) )
   {
      stdOut->writeString("Can't create " + options.m_sTargetDir );
      return false;
   }

}

bool transferModules( Options &options )
{
   ModuleLoader ml;

   ml.alwaysRecomp( true );
   ml.saveModules( false );

   if( options.m_sEncoding != "" )
      ml.sourceEncoding( options.m_sEncoding );

   // prepare the load path.
   if( options.m_sLoadPath != "" )
   {
      ml.addSearchPath( options.m_sLoadPath );
   }
   else
   {
      // See if we have a FALCON_LOAD_PATH envvar
      String retVal;
      if ( Sys::_getEnv( "FALCON_LOAD_PATH", retVal ) )
      {
         ml.addSearchPath( retVal );
      }
      else
      {
         ml.addSearchPath( "." );
      }
   }
   
   // add script path (always)
   Path scriptPath( options.m_sMainScript );
   if( scriptPath.getLocation() != "" )
      ml.addSearchPath( scriptPath.getLocation() );

   // load the application.
   Runtime rt( &ml );
   rt.loadFile( options.m_sMainScript );

   const ModuleVector* mv = rt.moduleVector();
   for( uint32 i = 0; i < mv->size(); i ++ )
   {
      Module *mod = mv->moduleAt(i);
      storeModule( options, mod );
   }

   return true;
}


int main( int argc, char *argv[] )
{
   Falcon::GetSystemEncoding( io_encoding );

   if ( io_encoding != "" )
   {
      Transcoder *trans = TranscoderFactory( io_encoding, 0, true );
      if ( trans == 0 )
      {
         stdOut = new StdOutStream();
         stdOut->writeString( "Unrecognized system encoding '" + io_encoding + "'; falling back to C.\n\n" );
         stdOut->flush();
      }
      else
      {
         stdOut = AddSystemEOL( TranscoderFactory( io_encoding, new StdOutStream, true ), true );
      }
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

   if ( ! options.m_sMainScript )
   {
      stdOut->writeString( "Nothing to do.\n\n" );
      return 0;
   }

   Engine::Init();

   // by default store the application in a subdirectory equal to the name of the
   // application.
   Path target( options.m_sMainScript );
   if( options.m_sTargetDir == "" )
   {
      options.m_sTargetDir = target.getFile();
   }

   //===============================================================
   // We need a runtime and a module loader to load all the modules.
   bool bResult;

   try
   {
      bResult = transferModules( options );

      if ( bResult )
      {
         if( ! options.m_bNoSysFile )
            bResult = transferSysFiles( options );
      }
   }
   catch( Error* err )
   {
      // We had a compile time problem, very probably
      bResult = false;
      stdOut->writeString( "Compilation error.\n" );
      stdOut->writeString( err->toString() );
      err->decref();
      stdOut->flush();
   }

   delete stdOut;

   Engine::Shutdown();
   
   return bResult ? 0 : 1;
}


/* end of falpack.cpp */

