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
#include "falpack_sys.h"
#include "options.h"
#include "utils.h"

namespace Falcon
{

bool transferSysFiles( Options &options, bool bJustScript )
{
   Path binpath, libpath;

   binpath.setLocation(
         options.m_sFalconBinDir != "" ? options.m_sFalconBinDir : FALCON_DEFAULT_BIN );
   // copy falcon or falrun
   if ( options.m_sRunner != "" )
      binpath.setFilename( options.m_sRunner );
   else
      binpath.setFilename( "falcon.exe" );

   libpath.setLocation(
         options.m_sFalconLibDir != "" ? options.m_sFalconLibDir : FALCON_DEFAULT_LIB );

   if ( ! bJustScript )
   {
   /*
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
      */
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


}

/* end of falpack_sys_unix.cpp */
