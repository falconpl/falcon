/*
   FALCON - The Falcon Programming Language.
   FILE: flcloader.cpp
   $Id: flcloader.cpp,v 1.18 2007/07/27 12:03:09 jonnymind Exp $

   Advanced module loader for Falcon programming language.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 22 2005
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Falcon language compiler - Advanced module loader - header
*/
#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <fasm/comp.h>
#include <falcon/gencode.h>
#include <falcon/genhasm.h>
#include <falcon/timestamp.h>
#include <falcon/sys.h>
#include <falcon/memory.h>
#include <falcon/stdstreams.h>
#include <falcon/stringstream.h>
#include <falcon/fstream.h>
#include <falcon/transcoding.h>
#include <falcon/flcloader.h>

namespace Falcon {

FlcLoader::FlcLoader( const String &path ):
      ModuleLoader( path ),
      m_alwaysRecomp( false ),
      m_compMemory( true ),
      m_viaAssembly( false ),
      m_saveModule( true ),
      m_sourceIsAssembly( false ),
      m_compileErrors(0),
      m_delayRaise( false ),
      m_saveMandatory( false )
{
   // we accept sources ( by default)
   m_acceptSources = true;
}


Stream *FlcLoader::openResource( const String &path, t_filetype type )
{
   Stream *in = ModuleLoader::openResource( path, type );

   if ( in != 0 && type == t_source && m_srcEncoding != "" )
   {
      // set input encoding
      Stream *inputStream = TranscoderFactory( m_srcEncoding, in, true );
      if( inputStream != 0 )
         return inputStream;
   }

   return in;
}


Module *FlcLoader::loadSource( const String &file )
{
   if ( ! m_alwaysRecomp )
   {
      FileStat fs_module;
      FileStat fs_source;
      // do we have a module?

      int32 dotpos = file.rfind( "." );
      int32 slashpos = file.rfind( "/" );
      String mod_name;

      // have we got a . before a /?
      if ( dotpos > slashpos )
         mod_name = file.subString( 0, dotpos ) + ".fam";
      else
         mod_name = file + ".fam";

      if ( Sys::fal_stats( mod_name, fs_module ) && Sys::fal_stats( file, fs_source ) )
      {
         if ( fs_module.m_mtime->compare( *fs_source.m_mtime ) > 0 )
         {
            return loadModule( mod_name );
         }
      }
   }

   // Ok, if we're here we have to load the source.
	// use the base load source routine
	Module *module = ModuleLoader::loadSource( file );

	// if the base load source worked, save the result (if configured to do so).
   if ( module != 0 &&  m_saveModule )
   {
      String target;
      int dotpos = file.rfind( "." );
      if ( dotpos > 0 ) // accepts .something
         target = file.subString( 0, dotpos ) + ".fam";
      else
         target = file + ".fam";

      FileStream *temp_binary;
      temp_binary = new FileStream;
      temp_binary->create( target, Falcon::FileStream::e_aReadOnly | Falcon::FileStream::e_aUserWrite );

      if ( ! temp_binary->good() || ! module->save( temp_binary ) )
      {
         if ( m_saveMandatory )
         {
            raiseError( e_file_output, target );
            delete temp_binary;

            delete module;
            return 0;
         }
      }

      delete temp_binary;
   }

   return module;
}


Module *FlcLoader::loadSource( Stream *fin )
{
   Module *module;
   bool bInAssembly;
   bool bDelFin = false;
   m_compileErrors = 0;

   // the temporary binary file for the pre-generated module
   Stream *temp_binary;

   if ( m_sourceIsAssembly )
   {
      bInAssembly = true;
   }
   else
   {
      module = new Module();
      bInAssembly = false;

      Compiler compiler( module, fin );
      compiler.delayRaise( m_delayRaise );

      compiler.errorHandler( m_errhand );
      if( ! compiler.compile() ) {
         m_compileErrors = (uint32) compiler.errors();
         delete module;
         return 0;
      }

      // we have compiled it. Now we need a file or a memory stream for
      // saving data.
      if ( m_compMemory ) {
         temp_binary = new StringStream;
      }
      else {
         String tempFileName;
         Sys::_tempName( tempFileName );
         FileStream *tb= new FileStream();
         tb->create( tempFileName + "_1", Falcon::FileStream::e_aReadOnly | Falcon::FileStream::e_aUserWrite );
         temp_binary = tb;
         if ( ! temp_binary->good() )
         {
            raiseError( e_file_output, tempFileName );
            delete temp_binary;
            delete module;
            return 0;
         }
      }

      if ( m_viaAssembly )
      {
         bInAssembly = true;
         // we'll turn fin into the binary module
         // so we have to delete it.
         bDelFin = true;
         fin = temp_binary;

         // as the output stream will be a text, use a text ranscoder.
         temp_binary = DefaultTextTranscoder( temp_binary, true );

         // generate.
         GenHAsm hasm( temp_binary );
         hasm.generatePrologue( compiler.module() );
         hasm.generate( compiler.sourceTree() );

         // the module is going to be destroyed anyhow
         // as we got to assemble it
         delete module;

         // now prepare to assemble the thing.
         temp_binary->seekBegin(0);
      }
      else {
         GenCode codeOut( temp_binary );
         codeOut.generate( compiler.sourceTree() );
         module->setLineInfo( codeOut.extractLineInfo() );
      }
   }


   // have we to assemble the module?
   if ( bInAssembly )
   {
      if ( m_compMemory ) {
         temp_binary = new StringStream;
      }
      else {
         String tempFileName;
         Sys::_tempName( tempFileName );
         FileStream *tb= new FileStream();
         tb->create( tempFileName + "_2", Falcon::FileStream::e_aReadOnly | Falcon::FileStream::e_aUserWrite );
         temp_binary = tb;
         if ( ! temp_binary->good() )
         {
            raiseError( e_file_output, tempFileName );
            if( bDelFin )
               delete fin;
            delete temp_binary;
            return 0;
         }
      }

      module = new Module();

      AsmCompiler fasm( module, fin, temp_binary );
      fasm.errorHandler( m_errhand );
      if ( ! fasm.compile() ) {
         m_compileErrors = (uint32) fasm.errors();
         delete module;
         if( bDelFin )
            delete fin;
         delete temp_binary;
         return 0;
      }
   }


   // import the binary stream in the module;
   uint64 len = temp_binary->seekEnd( 0 );
   Falcon::byte *code = (Falcon::byte *) memAlloc( (uint32) len );
   module->code( code );
   module->codeSize( (uint32) len );
   temp_binary->seekBegin( 0 );
   temp_binary->read( code, (int32) len );

   delete temp_binary;
   if( bDelFin )
      delete fin;

   if ( module != 0 )
      module->addMain();

   return module;
}

}


/* end of flcloader.cpp */
