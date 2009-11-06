/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.cpp

   Falcon compiler and interpreter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 10 Sept 2004 13:15:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   This is the falcon command line.

   Consider this a relatively complex example of an embedding application.
*/
#include "falcon.h"
#include "options.h"

#include <falcon/sys.h>
#include <falcon/genhasm.h>
#include <falcon/gentree.h>
#include <falcon/gencode.h>
#include <iostream>
#include <stdio.h>

#ifndef NDEBUG
#include <falcon/trace.h>
#endif

using namespace Falcon;
using namespace std;

/************************************************
   Falcon interpreter/compiler application
*************************************************/

AppFalcon::AppFalcon():
   m_exitval(0),
   m_errors(0),
   m_script_pos( 0 )
{
   // Install a void ctrl-c handler (let ctrl-c to kill this app)
   Sys::_dummy_ctrl_c_handler();

   // Prepare the Falcon engine to start.
   Engine::Init();
}

AppFalcon::~AppFalcon()
{
   // turn off the Falcon engine
   Engine::Shutdown();
}

//=================================================
// Utilities


void AppFalcon::terminate()
{
   if ( m_errors > 0 )
   {
      cerr << "falcon: exiting with ";
      if ( m_errors > 1 )
         cerr << m_errors << " errors." << endl;
      else
         cerr << "1 error." << endl;
   }

   if ( m_options.wait_after )
   {
      cout << "Press <ENTER> to terminate" << endl;
      getchar();
   }
}


String AppFalcon::getLoadPath()
{
   // should we ignore system paths?
   if ( m_options.ignore_syspath )
   {
      return m_options.load_path;
   }

   String envpath;
   String path = m_options.load_path;

   if( Sys::_getEnv ( "FALCON_LOAD_PATH", envpath ) )
   {
      if ( path.size() != 0 )
         path += ";";
      path += envpath;
   }

   // Otherwise, get the load path.
   if ( path.size() > 0)
      return  path + ";" + FALCON_DEFAULT_LOAD_PATH;
   else
      return FALCON_DEFAULT_LOAD_PATH;
}


String AppFalcon::getSrcEncoding()
{
   if ( m_options.source_encoding != "" )
      return m_options.source_encoding;

   if ( m_options.io_encoding != "" )
      return m_options.io_encoding;

   String envenc;
   if ( Sys::_getEnv ( "FALCON_SRC_ENCODING", envenc ) )
      return envenc;

   if ( Sys::_getEnv ( "FALCON_VM_ENCODING", envenc ) )
      return envenc;

   if ( GetSystemEncoding ( envenc ) )
      return envenc;

   // we failed.
   return "C";
}


String AppFalcon::getIoEncoding()
{
   // I/O encoding and source encoding priority is reversed here.
   if ( m_options.io_encoding != "" )
      return m_options.io_encoding;

   if ( m_options.source_encoding != "" )
      return m_options.source_encoding;

   String ret;
   if ( Sys::_getEnv ( "FALCON_VM_ENCODING", ret ) )
      return ret;

   if( GetSystemEncoding ( ret ) )
      return ret;

   return "C";
}


void AppFalcon::applyDirectives ( Compiler &compiler )
{
   ListElement *dliter = m_options.directives.begin();
   while ( dliter != 0 )
   {
      String &directive = * ( ( String * ) dliter->data() );
      // find "="
      uint32 pos = directive.find ( "=" );
      if ( pos == String::npos )
      {
         throw String( "directive not in <directive>=<value> syntax: \"" + directive + "\"" );
      }

      //split the directive
      String dirname ( directive, 0, pos );
      String dirvalue ( directive, pos + 1 );
      dirname.trim();
      dirvalue.trim();

      // is the value a number?
      int64 number;
      bool result;
      if ( dirvalue.parseInt ( number ) )
         result = compiler.setDirective ( dirname, number );
      else
         result = compiler.setDirective ( dirname, dirvalue );

      if ( ! result )
      {
         throw String( "invalid directive or value: \"" + directive + "\"" );
      }

      dliter = dliter->next();
   }
}


void AppFalcon::applyConstants ( Compiler &compiler )
{
   ListElement *dliter = m_options.defines.begin();
   while ( dliter != 0 )
   {
      String &directive = * ( ( String * ) dliter->data() );
      // find "="
      uint32 pos = directive.find ( "=" );
      if ( pos == String::npos )
      {
         throw String( "constant not in <directive>=<value> syntax: \"" + directive + "\"" );
      }

      //split the directive
      String dirname ( directive, 0, pos );
      String dirvalue ( directive, pos + 1 );
      dirname.trim();
      dirvalue.trim();

      // is the value a number?
      int64 number;
      if ( dirvalue.parseInt ( number ) )
         compiler.addIntConstant( dirname, number );
      else {
         compiler.addStringConstant( dirname, dirvalue );
      }

      dliter = dliter->next();
   }
}


bool AppFalcon::setup( int argc, char* argv[] )
{
   m_argc = argc;
   m_argv = argv;
   m_script_pos = argc;
   m_options.parse( argc, argv, m_script_pos );

   //=======================================
   // Check parameters && settings

   if ( ! m_options.wasJustInfo() )
   {
      String srcEncoding = getSrcEncoding();
      if ( srcEncoding != "" )
      {
         Transcoder *tcin = TranscoderFactory ( srcEncoding, 0, false );
         if ( tcin == 0 )
            throw String( "unrecognized encoding '" + srcEncoding + "'." );
         delete tcin;
      }

      String ioEncoding = getIoEncoding();
      if ( ioEncoding != "" )
      {
         Transcoder *tcin = TranscoderFactory ( ioEncoding, 0, false );
         if ( tcin == 0 )
            throw String( "unrecognized encoding '" + ioEncoding + "'." );
         delete tcin;
      }

      Engine::setEncodings( srcEncoding, ioEncoding );
#ifndef NDEBUG
      if ( m_options.trace_file != "" )
      {
         AutoCString trace_file(m_options.trace_file);
         TRACE_ON( trace_file.c_str() );
      }
#endif
   }


   return ! m_options.wasJustInfo();
}


void AppFalcon::readyStreams()
{
   // Function wide statics must be created here, as we may be making memory accounting later on.
   String ioEncoding = getIoEncoding();

   // change stdandard streams to fit needs
   if ( ioEncoding != "" && ioEncoding != "C" )
   {
      Transcoder* tcin = TranscoderFactory ( ioEncoding, new StreamBuffer( new StdInStream ), true );
      m_stdIn = AddSystemEOL ( tcin );
      Transcoder *tcout = TranscoderFactory ( ioEncoding, new StreamBuffer( new StdOutStream ), true );
      m_stdOut = AddSystemEOL ( tcout );
      Transcoder *tcerr = TranscoderFactory ( ioEncoding, new StreamBuffer( new StdErrStream ), true );
      m_stdErr = AddSystemEOL ( tcerr );
   }
   else
   {
      m_stdIn = AddSystemEOL ( new StreamBuffer( new StdInStream ) );
      m_stdOut = AddSystemEOL ( new StreamBuffer( new StdOutStream ) );
      m_stdErr = AddSystemEOL ( new StreamBuffer( new StdErrStream ) );
   }
}


Stream* AppFalcon::openOutputStream( const String &ext )
{
   Stream* out;

   if ( m_options.output == "-" )
   {
      out = new StdOutStream;
   }
   else if ( m_options.output == "" )
   {
      // if input has a name, try to open there.
      if ( m_options.input != "" && m_options.input != "-" )
      {
         #ifdef WIN32
            Path::winToUri( m_options.input );
         #endif
         URI uri_input( m_options.input );
         uri_input.pathElement().setFilename( uri_input.pathElement().getFile() +
            "." + ext );
         FileStream* fout = new FileStream;
         if ( ! fout->create( uri_input.get(), BaseFileStream::e_aUserRead | BaseFileStream::e_aUserWrite ) )
         {
            delete fout;
            throw String( "can't open output file '"+ uri_input.get() +"'" );
         }
         out = fout;
      }
      else {
         // no input and no output; output on stdout
         out = new StdOutStream;
      }
   }
   else
   {
      FileStream* fout = new FileStream;
      if ( ! fout->create( m_options.output, BaseFileStream::e_aUserRead | BaseFileStream::e_aUserWrite ) )
      {
         delete fout;
         throw String( "can't open output file '"+ m_options.output +"'" );
      }
      out = fout;
   }

   return out;
}


Module* AppFalcon::loadInput( ModuleLoader &ml )
{
   ml.sourceEncoding( getSrcEncoding() );
   Module* mod;

   if ( m_options.input != "" && m_options.input != "-" )
   {
      #ifdef WIN32
         Path::winToUri( m_options.input );
      #endif
      mod = ml.loadFile( m_options.input, ModuleLoader::t_none, true );
   }
   else
   {
      String ioEncoding = getSrcEncoding();
      Transcoder *tcin = TranscoderFactory ( ioEncoding == "" ? "C" : ioEncoding,
                                             new StreamBuffer( new StdInStream), true );
      mod = ml.loadSource( AddSystemEOL( tcin ), "<stdin>", "stdin" );
      delete tcin;
   }

   return mod;
}


void AppFalcon::compileTLTable()
{
   ModuleLoader ml;
   // the load path is not relevant, as we load by file name or stream
   // apply options
   ml.compileTemplate( m_options.parse_ftd );
   ml.saveModules( false );
   ml.alwaysRecomp( true );

   // will throw Error* on failure
   Module* mod = loadInput( ml );

   // try to open the oputput stream.
   Stream* out = 0;
   try
   {
       String ioEncoding = getIoEncoding();
       out = AddSystemEOL(
            TranscoderFactory ( ioEncoding == "" ? "C" : ioEncoding,
                  openOutputStream ( "temp.ftt" ), true ) );
      // Ok, we have the output stream.
      if ( ! mod->saveTableTemplate( out, ioEncoding == "" ? "C" : ioEncoding ) )
         throw String( "can't write on output stream." );
   }
   catch( ... )
   {
      delete out;
      mod->decref();
      throw;
   }

   delete out;
   mod->decref();
}


void AppFalcon::generateAssembly()
{
   ModuleLoader ml;
   // the load path is not relevant, as we load by file name or stream
   // apply options
   ml.compileTemplate( m_options.parse_ftd );
   ml.saveModules( false );
   ml.alwaysRecomp( true );

   // will throw Error* on failure
   Module* mod = loadInput( ml );

   // try to open the oputput stream.
   Stream* out = 0;

   try
   {
      String ioEncoding = getIoEncoding();
      out = AddSystemEOL(
         TranscoderFactory ( ioEncoding == "" ? "C" : ioEncoding,
            openOutputStream( "fas" ), true ) );

      // Ok, we have the output stream.
      GenHAsm gasm(out);
      gasm.generatePrologue( mod );
      gasm.generate( ml.compiler().sourceTree() );
      if ( ! out->good() )
         throw String( "can't write on output stream." );
   }
   catch( const String & )
   {
      delete out;
      mod->decref();
      throw;
   }

   delete out;
   mod->decref();
}


void AppFalcon::generateTree()
{
   Compiler comp;
   // the load path is not relevant, as we load by file name or stream
   // apply options
   Stream* in = 0;

   // will throw Error* on failure
   if ( m_options.input != "" && m_options.input != "-" )
   {
      #ifdef WIN32
         Path::winToUri( m_options.input );
      #endif
      FileStream* fs = new FileStream();
      fs->open( m_options.input );
      in = fs;
   }
   else
   {
      String ioEncoding = getSrcEncoding();
      Transcoder *tcin = TranscoderFactory ( ioEncoding == "" ? "C" : ioEncoding,
                                             new StreamBuffer( new StdInStream), true );
      in = tcin;
   }

   // try to open the oputput stream.
   Stream* out = 0;
   Module *mod = new Module;

   try
   {
      comp.compile( mod, in );

      String ioEncoding = getIoEncoding();
      out = AddSystemEOL(
         TranscoderFactory ( ioEncoding == "" ? "C" : ioEncoding,
            openOutputStream ( "fr" ), true ) );

      // Ok, we have the output stream.
      GenTree gtree(out);
      gtree.generate( comp.sourceTree() );
      if ( ! out->good() )
         throw String( "can't write on output stream." );
   }
   catch( const String & )
   {
      delete in;
      delete out;
      mod->decref();
      throw;
   }

   delete in;
   delete out;
   mod->decref();
}


void AppFalcon::buildModule()
{
   ModuleLoader ml;
   // the load path is not relevant, as we load by file name or stream
   // apply options
   ml.compileTemplate( m_options.parse_ftd );
   ml.saveModules( false );
   ml.alwaysRecomp( true );

   // will throw Error* on failure
   Module* mod = loadInput( ml );

   // try to open the oputput stream.
   Stream* out = 0;

   try
   {
      // binary
      out = openOutputStream ( "fam" );

      // Ok, we have the output stream.
      GenCode gcode(mod);
      gcode.generate( ml.compiler().sourceTree() );
      if ( ! mod->save( out ) )
         throw String( "can't write on output stream." );
   }
   catch( const String & )
   {
      delete out;
      mod->decref();
      throw;
   }

   delete out;
   mod->decref();
}


void AppFalcon::makeInteractive()
{
   m_options.version();
   readyStreams();
   IntMode ic( this );
   ic.run();
}

void AppFalcon::prepareLoader( ModuleLoader &ml )
{
   // 1. Ready the module loader
   ModuleLoader *modLoader = &ml;
   ml.setSearchPath( getLoadPath() );

   // adds also the input path.
   if ( m_options.input != "" && m_options.input != "-" )
   {
      URI in_uri( m_options.input );
      in_uri.pathElement().setFilename("");
      // empty path? -- add current directory (may be removed from defaults)
      if ( in_uri.get() == "" )
         modLoader->addSearchPath ( "." );
      else
         modLoader->addSearchPath ( in_uri.get() );
   }

   // set the module preferred language; ok also if default ("") is used
   modLoader->setLanguage ( m_options.module_language );
   applyDirectives( modLoader->compiler() );
   applyConstants( modLoader->compiler() );

   // save the main module also if compile only option is set
   modLoader->saveModules ( m_options.save_modules );
   modLoader->compileInMemory ( m_options.comp_memory );
   modLoader->alwaysRecomp ( m_options.force_recomp );
   modLoader->sourceEncoding ( getSrcEncoding() );
   // normally, save is not mandatory, unless we compile them our own
   // should be the default, but we reset it.
   modLoader->saveMandatory ( false );

   // should we forcefully consider input as ftd?
   modLoader->compileTemplate ( m_options.parse_ftd );

   Engine::setSearchPath( modLoader->getSearchPath() );
}


void AppFalcon::runModule()
{
   ModuleLoader ml;
   prepareLoader( ml );

   // Create the runtime using the given module loader.
   Runtime runtime( &ml );

   // now that we have the main module, inject other requested modules
   ListElement *pliter = m_options.preloaded.begin();
   while ( pliter != 0 )
   {
      Module *module = ml.loadName ( * ( ( String * ) pliter->data() ) );
      runtime.addModule( module );

      // abandon our reference to the injected module
      module->decref();

      pliter = pliter->next();
   }

   // then add the main module
   Module* mainMod = loadInput(ml);
   runtime.addModule( mainMod );

   // abandon our reference to the main module
   mainMod->decref();

   //===========================================
   // Prepare the virtual machine
   //
   VMachineWrapper vmachine;

   //redirect the VM streams to ours.
   // The machine takes ownership of the streams, so they won't be useable anymore
   // after the machine destruction.
   readyStreams();
   vmachine->stdIn( m_stdIn );
   vmachine->stdOut( m_stdOut );
   vmachine->stdErr( m_stdErr );
   // I have given real process streams to the vm
   vmachine->hasProcessStreams( true );

   // push the core module
   // we know we're not launching the core module.
   vmachine->launchAtLink( false );
   Module* core = core_module_init();
   #ifdef NDEBUG
      vmachine->link ( core );
   #else
      LiveModule *res = vmachine->link ( core );
      fassert ( res != 0 ); // should not fail
   #endif
   core->decref();

   // prepare environment
   Item *item_args = vmachine->findGlobalItem ( "args" );
   fassert ( item_args != 0 );
   CoreArray *args = new CoreArray ( m_argc - m_script_pos );

   String ioEncoding = getIoEncoding();
   for ( int ap = m_script_pos; ap < m_argc; ap ++ )
   {
      CoreString *cs = new CoreString;
      if ( ! TranscodeFromString ( m_argv[ap], ioEncoding, *cs ) )
      {
         cs->bufferize ( m_argv[ap] );
      }

      args->append ( cs );
   }

   item_args->setArray ( args );

   Item *script_name = vmachine->findGlobalItem ( "scriptName" );
   fassert ( script_name != 0 );
   *script_name = new CoreString ( mainMod->name() );

   Item *script_path = vmachine->findGlobalItem ( "scriptPath" );
   fassert ( script_path != 0 );
   *script_path = new CoreString ( mainMod->path() );

   // Link the runtime in the VM.
   // We'll be running the modules as we link them in.
   vmachine->launchAtLink( true );
   if ( vmachine->link( &runtime ) )
   {
      vmachine->launch();

      if ( vmachine->regA().isInteger() )
         exitval( ( int32 ) vmachine->regA().asInteger() );
   }
}

void AppFalcon::run()
{
   // record the memory now -- we're gonna create the streams that will be handed to the VM
   size_t memory = gcMemAllocated();
   int32 items = memPool->allocatedItems();

   // determine the operation mode
   if( m_options.compile_tltable )
      compileTLTable();
   else if ( m_options.assemble_out )
      generateAssembly();
   else if ( m_options.tree_out )
      generateTree();
   else if ( m_options.compile_only )
      buildModule();
   else if ( m_options.interactive )
      makeInteractive();
   else
      runModule();

   memPool->performGC();

   if ( m_options.check_memory )
   {
      // be sure we have reclaimed all what's possible to reclaim.
      size_t mem2 = gcMemAllocated();
      int32 items2 = memPool->allocatedItems();
      cout << "===============================================================" << std::endl;
      cout << "                 M E M O R Y    R E P O R T" << std::endl;
      cout << "===============================================================" << std::endl;
      cout << " Unbalanced memory: " << mem2 - memory << endl;
      cout << " Unbalanced items : " << items2 - items << endl;
      cout << "===============================================================" << std::endl;
   }
}

//===========================================
// Main Routine
//===========================================

int main ( int argc, char *argv[] )
{
   AppFalcon falcon;
   StdErrStream serr;

   try {
      if ( falcon.setup( argc, argv ) )
         falcon.run();
   }
   catch ( const String &fatal_error )
   {
      serr.writeString( "falcon: FATAL - " + fatal_error + "\n" );
      falcon.exitval( 1 );
   }
   catch ( Error *err )
   {
      String temp;
      err->toString( temp );
      serr.writeString( "falcon: FATAL - Program terminated with error.\n" );
      serr.writeString( temp + "\n" );
      err->decref();
      falcon.exitval( 1 );
   }

   falcon.terminate();
   return falcon.exitval();
}

/* end of falcon.cpp */
