/*
   FALCON - The Falcon Programming Language.
   FILE: falpack_sys_unix.cpp

   System specific extensions for Falcon
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jan 2010 11:29:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>
#include "falpack_sys.h"
#include "options.h"
#include "utils.h"

#include <stdio.h>

namespace Falcon
{

bool transferSysFiles( Options &options, bool bJustScript )
{
   Path binpath, libpath;

   binpath.setLocation(
         options.m_sFalconBinDir != "" ? options.m_sFalconBinDir : FALCON_DEFAULT_BIN );
   binpath.setFilename( "falcon" );

   Path runnerPath( options.m_sRunner );
   if( runnerPath.get() != "" && runnerPath.isValid() )
   {
      binpath.setFilename( runnerPath.getFilename() );
   }

   libpath.setLocation(
         options.m_sFalconLibDir != "" ? options.m_sFalconLibDir : FALCON_DEFAULT_LIB );

   if ( ! bJustScript )
   {
      Path tgtpath( options.m_sTargetDir + "/" + options.m_sSystemRoot +"/" );

      libpath.setFile( "libfalcon_engine" );
      libpath.setExtension( DllLoader::dllExt() );

      tgtpath.setFilename( binpath.getFilename() );
      if( ! copyFile( binpath.get(), tgtpath.get() ) )
      {
         warning( "Can't copy system file \"" + binpath.get()
               + "\" into target path \""+ tgtpath.get()+ "\"" );
         // but continue
      }

      Sys::fal_chmod( tgtpath.get(), 0755 );

      tgtpath.setFilename( libpath.getFilename() );
      if( ! copyFile( libpath.get(), tgtpath.get() ) )
      {
         warning( "Can't copy system file \"" + libpath.get()
               + "\" into target path \""+ tgtpath.get()+ "\"" );
         // but continue
      }
   }

   // now create the startup script
   Path mainScriptPath( options.m_sMainScript );
   Path scriptPath( options.m_sTargetDir + "/" + mainScriptPath.getFile() + ".sh" );

   message( "Creating startup script \"" + scriptPath.get() + "\"" );
   FileStream startScript;
   if( ! startScript.create( scriptPath.get(), (BaseFileStream::t_attributes) 0755 ) )
   {
      error( "Can't create startup script \"" + scriptPath.get() + "\"" );
      return false;
   }

   startScript.writeString( "#!/bin/sh\n" );
   startScript.writeString( "CURDIR=`dirname $0`\n" );
   startScript.writeString( "cd \"$CURDIR\"\n" );
   if( bJustScript )
   {
      if ( runnerPath.isAbsolute() )
         startScript.writeString( "    "+ runnerPath.get() + " \\\n" );
      else
         startScript.writeString( "    "+ binpath.getFilename() + " \\\n" );
   }
   else
   {
      startScript.writeString( "LD_LIBRARY_PATH=\"" + options.m_sSystemRoot + "\" \\\n" );
      startScript.writeString( "   \""+options.m_sSystemRoot + "/" + binpath.getFilename() + "\" \\\n" );
      startScript.writeString( "    -L \"" + options.m_sSystemRoot +";.\" \\\n" );
   }

   // we need to discard the extension, so that the runner decides how to run the program.
   Path scriptName( options.m_sMainScript );
   startScript.writeString( "    \"" + scriptName.getFile() +"\" $*" );

   startScript.flush();

   return true;
}


bool copyDynlibs( Options& options, const String& modpath, const std::vector<String>& dynlibs )
{
   // On unix, the thing is a bit more complex.
   // We need to find the right library via ldd and copy
   // it.
   AutoCString command( "ldd " + modpath );
   FILE* ldin = popen( command.c_str(), "r" );

   if( ldin == NULL )
   {
      warning( "Cannot copy required dynlibs for " + modpath + " (Cannot start ldd)" );
      return false;
   }

   char buffer[4096];
   String source;
   // parallel vector
   uint32 size = dynlibs.size();

   // file copy destination is always the root of the system path
   // where we set the LD_LIBRARY_PATH of the startup script.
   Path targetPath;
   targetPath.setFullLocation( options.m_sTargetDir + "/" + options.m_sSystemRoot );

   // find our module
   // we must read entirely LDD output, or we may fail to close it via pclose.
   while ( fgets( buffer, 4096, ldin ) )
   {
      // still something to be found?
      if( size > 0 )
      {
         String line( buffer );

         // search all the libs
         for ( uint32 i = 0; i < dynlibs.size(); ++i )
         {
            if( line.wildcardMatch( "*" + dynlibs[i] + "*."+DllLoader::dllExt() + "*") )
            {
               bool done = false;
               uint32 pos = line.find( "=>" );
               uint32 pos1 = line.find( "(", pos );
               if( pos != String::npos && pos1 != String::npos )
               {
                  String srcLib = line.subString( pos+2, pos1 );
                  srcLib.trim();
                  if( srcLib != "" )
                  {
                     Path sourceLib( srcLib );
                     targetPath.setFilename( sourceLib.getFilename() );
                     done = copyFile( srcLib, targetPath.get() );
                  }
               }

               if( ! done )
               {
                  warning( "Cannot extract possible plugin from ldd line " + line );
               }

               size--;
            }
         }
      }

      // do we have a <lib>?name[-].*.<libext> pattern in the source list?
   }

   if ( pclose(ldin) == -1 )
   {
      warning( "Can't close ldd process for module " + modpath );
   }

   if( size >  0 )
   {
      warning( "Can't resolve all the dynlibs for module " + modpath );
   }

   return true;
}


}

/* end of falpack_sys_unix.cpp */
