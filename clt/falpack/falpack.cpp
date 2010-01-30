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
#include "utils.h"

using namespace Falcon;

String load_path;
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
   stdOut->writeString( "   -e <enc>    Source files encoding.\n" );
   stdOut->writeString( "   -h, -?      This help.\n" );
   stdOut->writeString( "   -L <dir>    Redefine FALCON_LOAD_PATH\n" );
   stdOut->writeString( "   -M          Pack also pre-compiled modules.\n" );
   stdOut->writeString( "   -P <dir>    Store non-application libraries in this subdirectory\n" );
   stdOut->writeString( "   -r          Use falrun instead of falcon as runner\n" );
   stdOut->writeString( "   -s          Strip sources.\n" );
   stdOut->writeString( "   -S          Do not store system files (engine + runner)\n" );
   stdOut->writeString( "   -v          Prints version and exit.\n" );
   stdOut->writeString( "   -V          Verbose mode.\n" );
   stdOut->writeString( "\n" );
   stdOut->flush();
}


bool copyFile( const String& source, const String& dest )
{
   message( String("Copying ").A( source ).A(" => ").A( dest ) );

   // NOTE: streams are closed by the destructor.
   FileStream instream, outstream;

   instream.open( source, ::Falcon::BaseFileStream::e_omReadOnly );
   if ( ! instream.good() )
   {
      return false;
   }

   outstream.create( dest, (Falcon::BaseFileStream::t_attributes) 0644 );
   if ( ! outstream.good() )
   {
      return false;
   }

   byte buffer[8192];
   int count = 0;
   while( ( count = instream.read( buffer, 8192) ) > 0 )
   {
      if ( outstream.write( buffer, count ) < 0 )
      {
         return false;
      }
   }

   return true;
}

bool copyAllResources( Options& options, const Path& from, const Path& tgtPath )
{
   // do we have an extension filter?
   bool bHasExt = from.getExtension() !=  "";

   VFSProvider* file = Engine::getVFS("file");
   if( file == 0 )
   {
      error( "Can't find FILE resource" );
      return false;
   }

   DirEntry *entry = file->openDir( from.getLocation() );
   if( entry == 0 )
   {
      warning( "Can't open directory " + from.getLocation() );
      return false;
   }

   String fname;
   while( entry->read( fname ) )
   {
      if( fname == ".." || fname == "." )
      {
         continue;
      }

      FileStat fs;
      if ( ! Sys::fal_stats( from.getLocation() + "/" + fname, fs ) )
      {
         continue;
      }

      if ( fs.m_type == FileStat::t_normal || fs.m_type == FileStat::t_link )
      {
         // do we filter the extension?
         if( bHasExt )
         {
            if ( ! fname.endsWith( "." + from.getExtension(), true ) )
            {
               continue;
            }
         }

         // TODO: Jail resources under modpath
         if ( ! copyFile( from.getLocation() + "/" + fname, tgtPath.getLocation() + "/" + fname ) )
         {
            warning( "Cannot copy resource " +
                  from.getLocation() + "/" + fname
                  + " into "
                  + tgtPath.getLocation() + "/" + fname );
            entry->close();
            delete entry;
            return false;
         }

         /*
         // descend
         Path nfrom( from );
         nfrom.setLocation( from.getLocation() + "/" + fname );
         if( ! copyAllResources( options, nfrom, modPath, tgtPath ) )
         {
            return false;
         }
         */
      }


   }

   entry->close();
   delete entry;

   return true;
}


bool copyResource( Options& options, const String& resource, const Path& modPath, const Path& tgtPath )
{
   message( "Storing resource " + resource );
   Path resPath( resource );

   Path modResPath( modPath );
   Path tgtResPath( tgtPath );

   if( resPath.isAbsolute() )
   {
      warning( "Resource " + resource + " has an absolute path." );
      modResPath.setLocation( modPath.getLocation() + resPath.getLocation() );
      tgtResPath.setLocation( tgtPath.getLocation() + resPath.getLocation() );
   }
   else
   {
      modResPath.setLocation( modPath.getLocation() +"/"+ resPath.getLocation() );
      tgtResPath.setLocation( tgtPath.getLocation() +"/"+ resPath.getLocation() );
   }

   modResPath.setFilename( resPath.getFilename() );
   tgtResPath.setFilename( resPath.getFilename() );

   // create target path
   int32 fsStatus;
   if( ! Sys::fal_mkdir( tgtResPath.getLocation(), fsStatus, true ) )
   {
      warning( "Cannot create path " + tgtResPath.getLocation()
            + " for resource " + modResPath.get() );

      return false;
   }

   if( resPath.getFile() == "*" )
   {
     Path from( resPath );
     from.setLocation( modPath.getLocation() + "/" + from.getLocation() );
     if ( ! copyAllResources( options, from, tgtResPath ) )
     {
        return false;
     }
   }
   else
   {
      // TODO: Jail resources under modpath
      if ( ! copyFile( modResPath.get(), tgtResPath.get() ) )
      {
         warning( "Cannot copy resource " + modResPath.get() + " into " + tgtResPath.get() );
         return false;
      }
   }

   return true;
}

//TODO fill this per system
bool transferSysFiles( Options &options )
{

   return true;
}



bool storeModule( Options& options, Module* mod )
{
   // this is the base path for the module
   Path modPath( mod->path() );
   Path tgtPath;
   tgtPath.setLocation( options.m_sTargetDir );

   message( String("Processing module ").A( mod->path() ) );

   // strip the main script path from the module path.
   String modloc = modPath.getLocation();

   if ( modloc.find( options.m_sMainScriptPath ) == 0 )
   {
      // The thing came from below the main script.
      modloc = modloc.subString(options.m_sMainScriptPath.length() );
      if ( modloc != "" && modloc.getCharAt(0) == '/' )
      {
         modloc = modloc.subString(1);
      }
      tgtPath.setLocation( tgtPath.get() + "/" + modloc );
   }
   else
   {
      // if it's coming from somewhere else in the loadpath hierarcy,
      // we must store it below the topmost dir.
   }

   // store it
   int fsStatus;
   if ( ! Sys::fal_mkdir( tgtPath.getLocation(), fsStatus, true ) )
   {
      error( String("Can't create ") + tgtPath.getLocation() );
      return false;
   }

   tgtPath.setFilename( modPath.getFilename() );

   // should we store just sources, just fam or both?
   bool result;
   if( modPath.get() != ".fam" && modPath.get() != DllLoader::dllExt() )
   {
      // it's a source file.
      if ( ! options.m_bStripSources )
      {
         if( ! copyFile( modPath.get(), tgtPath.get() ) )
         {
            error( String("Can't copy \"") + modPath.get() + "\" into \"" +
                  tgtPath.get() + "\"" );
            return false;
         }
      }

      // should we save the fam?
      if( options.m_bPackFam )
      {
         tgtPath.setExtension("fam");
         FileStream famFile;

         if ( ! famFile.create( tgtPath.get(), (Falcon::BaseFileStream::t_attributes) 0644  )
             || ! mod->save(&famFile, false) )
         {
            error( "Can't create \"" + tgtPath.get() + "\"" );
            return false;
         }
      }
   }
   else
   {
      // just blindly copy everything else.
      if( ! copyFile( modPath.get(), tgtPath.get() ) )
      {
         error( "Can't copy \"" + modPath.get() + "\" into \"" + tgtPath.get() + "\"" );
         return false;
      }
   }

   // now copy .ftr files, if any.
   modPath.setExtension( "ftr" );
   FileStat ftrStat;

   if ( Sys::fal_stats( modPath.get(), ftrStat ) )
   {
      message( "Copying translation file " + modPath.get() );

      tgtPath.setExtension( "ftr" );
      // just blindly copy everything else.
      if( ! copyFile( modPath.get(), tgtPath.get() ) )
      {
         warning( "Can't copy \"" + modPath.get() + "\" into \"" + tgtPath.get() + "\"\n" );
      }
   }

   // and now, the resources.
   AttribMap* attributes =  mod->attributes();
   VarDef* resources;
   if( attributes != 0 && (resources = attributes->findAttrib("resources")) != 0 )
   {
      message( "Copying resources for module " + mod->path() );

      if( resources != 0 )
      {
         if ( ! resources->isString() || resources->asString()->size() == 0 )
         {
            warning( "Module \"" + mod->path() + " has an invalid \"resources\" attribute.\n" );
         }
         else
         {
            // split the resources in ";"
            String sAllRes = *resources->asString();
            uint32 pos = 0, pos1;
            while( (pos1 = sAllRes.find( ";", pos )) != String::npos )
            {
               String sRes = sAllRes.subString( pos, pos1 );
               sRes.trim();
               copyResource( options, sRes, modPath, tgtPath );
               pos = pos1+1;
            }

            String sRes = sAllRes.subString( pos );
            sRes.trim();
            copyResource( options, sRes, modPath, tgtPath );
         }
      }
   }

   return true;
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

   // add the main script path to the options, so that it can be stripped.
   options.m_sMainScriptPath = scriptPath.getLocation();

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
         stdErr = AddSystemEOL( TranscoderFactory( io_encoding, new StdOutStream, true ), true );
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

   setVerbose( options.m_bVerbose );

   if ( ! options.m_sMainScript )
   {
      stdOut->writeString( "falpack: Nothing to do.\n\n" );
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
      error( String( "Compilation error.\n" ) + err->toString() );
      err->decref();
   }

   delete stdOut;
   delete stdErr;

   Engine::Shutdown();
   
   return bResult ? 0 : 1;
}


/* end of falpack.cpp */

