/*
   FALCON - The Falcon Programming Language.
   FILE: falpack_sys_win.cpp

   System specific extensions for Falcon
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jan 2010 11:29:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>
#include <falcon/sys.h>
#include "falpack_sys.h"
#include "options.h"
#include "utils.h"

namespace Falcon
{

bool copyRuntime( const Path& binpath, const Path& tgtpath )
{
   message( "Searching VSx CRT in " + binpath.getFullLocation() );

   // open the binary path in search of "Microsoft.*.CRT"
   VFSProvider* provider = Engine::getVFS( "file" );
   fassert( provider != 0 );
   
   DirEntry* dir = provider->openDir( binpath.getFullLocation() );
   if( dir == 0 )
   {
      warning( "Can't search CRT in " + binpath.getFullLocation() );
      return false;
   }
   
   String fname;
   while( dir->read(fname) )
   {
      if( fname.wildcardMatch("Microsoft.*.CRT") )
      {
         // we're done with dir.
         delete dir;

         Path source( binpath.getFullLocation() + "/" + fname + "/");
         Path target( tgtpath.getFullLocation() + "/" + fname + "/");
         
         // first, create the target path
         int32 fsStatus;
         if( ! Sys::fal_mkdir( target.getFullLocation(), fsStatus, true ) )
         {
            warning( "Can't create CRT directory in " + target.getFullLocation() );
            return false;
         }

         // then copy everything inside it.
         DirEntry* crtdir = provider->openDir( source.getFullLocation() );
         if( crtdir == 0 )
         {
            warning( "Can't read source CRT directory " + source.getFullLocation() );
            return false;
         }

         //loop copying everything that's not a dir.
         String sFile;
         while( crtdir->read( sFile ) )
         {
            if( sFile.startsWith(".") )
               continue;

            source.setFilename( sFile );
            target.setFilename( sFile );
            if ( ! copyFile( source.get(), target.get() ) )
            {
               delete crtdir;
               warning( "Can't copy CRT file " + source.get() +  " into " 
                  + target.get() );
               return false;
            }
         }

         return true;
      }
   }

   delete dir;
   return false;
}
                 


bool transferSysFiles( Options &options, bool bJustScript )
{
   Path binpath, libpath;

   // Under windows, the binary path is usually stored in an envvar.

   String envpath;
   if ( ! Sys::_getEnv( "FALCON_BIN_PATH", envpath ) || envpath == "" )
      envpath = FALCON_DEFAULT_BIN;

   binpath.setFullLocation(
      options.m_sFalconBinDir != "" ? options.m_sFalconBinDir: envpath );
   // copy falcon or falrun
   if ( options.m_sRunner != "" )
      binpath.setFilename( options.m_sRunner );
   else
      binpath.setFilename( "falcon.exe" );

   // our dlls are in bin, under windows.
   libpath.setFullLocation(
         options.m_sFalconLibDir != "" ? options.m_sFalconLibDir : envpath );

   if ( ! bJustScript )
   {
      Path tgtpath( options.m_sTargetDir + "/" + options.m_sSystemRoot +"/" );

      libpath.setFile( "falcon_engine" );
      libpath.setExtension( DllLoader::dllExt() );

      tgtpath.setFilename( binpath.getFilename() );
      if( ! copyFile( binpath.get(), tgtpath.get() ) )
      {
         warning( "Can't copy system file \"" + binpath.get()
               + "\" into target path \""+ tgtpath.get()+ "\"" );
         // but continue
      }

      tgtpath.setFilename( libpath.getFilename() );
      if( ! copyFile( libpath.get(), tgtpath.get() ) )
      {
         warning( "Can't copy system file \"" + libpath.get()
               + "\" into target path \""+ tgtpath.get()+ "\"" );
         // but continue
      }

      // and now the visual C runtime, if any
      copyRuntime( binpath, tgtpath );
   }

   // now create the startup script
   Path mainScriptPath( options.m_sMainScript );
   Path scriptPath( options.m_sTargetDir + "/" + mainScriptPath.getFile() + ".bat" );

   message( "Creating startup script \"" + scriptPath.get() + "\"" );
   FileStream startScript;
   if( ! startScript.create( scriptPath.get(), (BaseFileStream::t_attributes) 0777 ) )
   {
      error( "Can't create startup script \"" + scriptPath.get() + "\"" );
      return false;
   }

   startScript.writeString( 
      "@ECHO OFF\r\n"
      "set OLD_DIR=%CD%\r\n"
      "set target_dir=%~dp0\r\n"
      "cd %target_dir%\r\n");
   
   if( bJustScript )
   {
      startScript.writeString( "\"" + binpath.getFilename() + "\" " );
   }
   else
   {
      startScript.writeString( "   \""+options.m_sSystemRoot + "\\" + binpath.getFilename() + "\" " );
      startScript.writeString( "    -L \"" + options.m_sSystemRoot +";.\" " );
   }

   // we need to discard the extension, so that the runner decides how to run the program.
   Path scriptName( options.m_sMainScript );
   startScript.writeString( " \"" + scriptName.getFile() +"\" \"%*\"\r\n" );
   
   startScript.writeString( "cd %OLD_DIR%\r\n" );
   startScript.flush();

   return true;
}


bool copyDynlibs( Options& options, const String& mp, const std::vector<String>& dynlibs )
{
   // On windows, the thing is simple. 
   // We must add the module named as <module>.dll besides the host module if it's binary, 
   // or in the Falcon system dir if it's a source module.

   Path modpath( mp );
   Path targetPath;
   String ext = modpath.getExtension();
   ext.lower();

   // binary module ?
   if( ext == DllLoader::dllExt() )
      targetPath.setFullLocation( options.m_sTargetDir + "/" + options.m_sSystemRoot );
   else
      targetPath = modpath;

   bool retval = true;
   for ( uint32 i = 0; i < dynlibs.size(); ++i )
   {
      modpath.setFilename( dynlibs[i] + "." + DllLoader::dllExt() );
      targetPath.setFilename( dynlibs[i] + "." +  DllLoader::dllExt() );

      if( ! copyFile( modpath.get(), targetPath.get() ) )
      {
         warning( "Cannot copy dynlib resource " + modpath.get() );
         retval = false;
      }
   }
   
   return retval;
}

}

/* end of falpack_sys_unix.cpp */
