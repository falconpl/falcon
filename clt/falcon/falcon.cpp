/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.cpp

   Falcon compiler and interpreter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven set 10 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   This is the falcon command line.

   Consider this a relatively complex example of an embedding application.
*/

#include <falcon/sys.h>
#include <falcon/setup.h>
#include <falcon/common.h>
#include <falcon/compiler.h>
#include <falcon/genhasm.h>
#include <falcon/gencode.h>
#include <falcon/gentree.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/modloader.h>
#include <falcon/flcloader.h>
#include <falcon/runtime.h>
#include <falcon/core_ext.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/memory.h>
#include <falcon/transcoding.h>
#include <falcon/stream.h>
#include <falcon/fstream.h>
#include <falcon/stringstream.h>
#include <falcon/stdstreams.h>
#include <falcon/deferrorhandler.h>
#include <fasm/comp.h>
#include <falcon/fassert.h>

#include "options.h"

using namespace Falcon;

/********************************************
   Static data used by Falcon command line
*********************************************/
/** Options for falcon command line */
static HOptions options;
Stream *stdOut;
Stream *stdErr;
Stream *stdIn;


/************************************************
   Functions used to account memory management
*************************************************/

//TODO: move in the engine and provide a better interface.
//TODO: Record the address of the caller.

static long s_allocatedMem = 0;
static long s_outBlocks = 0;
static long s_validAlloc = 1;

#define MEMBLOCK_DATA_COUNT 2
#define MEMBLOCK_SIZE (sizeof(long) * MEMBLOCK_DATA_COUNT)


static void *account_alloc( size_t size )
{
   if ( size == 0 )
      return 0;

	long *mem = (long *) malloc( size + MEMBLOCK_SIZE );
	mem[0] = (long) size;
	mem[1] = 0xFEDCBA98;
	s_allocatedMem += size;
   s_outBlocks++;
	return mem + MEMBLOCK_DATA_COUNT;
}

static void account_free( void *mem )
{
	if ( mem == 0 )
		return;

	long *block = (long *) mem;
	block = block - MEMBLOCK_DATA_COUNT;

	if ( block[1] != 0xFEDCBA98 ) {
		s_validAlloc = 0;
		return;
	}

	s_allocatedMem -= block[0];
   s_outBlocks--;

	free( block );
}

static void *account_realloc( void *mem, size_t size )
{
	long *block = (long *) mem;

	if ( mem != 0 )
	{
		block = block - MEMBLOCK_DATA_COUNT;
		if ( block[1] != 0xFEDCBA98 ) {
			s_validAlloc = 0;
			block = (long *) malloc( size + MEMBLOCK_SIZE );
		}
		else {
			s_allocatedMem -= block[0];
			block = (long *) realloc( block, size + MEMBLOCK_SIZE );
		}
	}
	else {
		block = (long *) malloc( size + MEMBLOCK_SIZE );
      s_outBlocks++;
	}

   if( size != 0 )
   {
	   s_allocatedMem += (long) size;
   }
   else {
      s_outBlocks--;
   }

	block[0] = (long) size;
	block[1] = 0xFEDCBA98;
	return block + MEMBLOCK_DATA_COUNT;
}


/************************************************
   Typical utility functions for command lines
*************************************************/

static void version()
{
   stdOut->writeString( "Falcon compiler and interpreter.\n" );
   stdOut->writeString( "Version " );
   stdOut->writeString( FALCON_VERSION " (" FALCON_VERSION_NAME ")" );
   stdOut->writeString( "\n" );
}

static void usage()
{
   stdOut->writeString( "Usage: falcon [options] file.fal [script options]\n" );
   stdOut->writeString( "\n" );
   stdOut->writeString( "Options:\n" );
   stdOut->writeString( "   -a          assemble the given module (a Falcon Assembly '.fas' file)\n" );
   stdOut->writeString( "   -c          compile only the given source\n" );
   stdOut->writeString( "   -C          Check for memory allocation correctness.\n" );
   stdOut->writeString( "   -D          Set directive (as <directive>=<value>).\n" );
   stdOut->writeString( "   -e <enc>    Set given encoding as default for VM I/O.\n" );
   stdOut->writeString( "   -E <enc>    Source files are in <enc> encoding (overrides -e)\n" );
   stdOut->writeString( "   -f          force recompilation of modules even when .fam are found\n" );
   stdOut->writeString( "   -h/-?       this help\n" );
   stdOut->writeString( "   -l <lang>   Set preferential language of loaded modules\n" );
   stdOut->writeString( "   -L <path>   set path for 'load' directive\n" );
   stdOut->writeString( "   -m          do NOT compile in memory (use temporary files)\n" );
   stdOut->writeString( "   -M          do NOT save the compiled modules in '.fam' files\n" );
   stdOut->writeString( "   -o <fn>     output to <fn> instead of [filename.xxx]\n" );
   stdOut->writeString( "   -p <module> preload (pump in) given module\n" );
   stdOut->writeString( "   -P          use load path also to find main module\n" );
   stdOut->writeString( "   -r          do NOT recompile sources to fulfil load directives\n" );
   stdOut->writeString( "   -s          compile via assembly\n" );
   stdOut->writeString( "   -S          produce an assembly output\n" );
   stdOut->writeString( "   -t          generate a syntactic tree (for logic debug)\n" );
   stdOut->writeString( "   -T          force input parsing as .ftd (template document)\n" );
   stdOut->writeString( "   -v          print copyright notice and version and exit\n" );
   stdOut->writeString( "   -w          Add an extra console wait after program exit\n" );
   stdOut->writeString( "   -x          execute a binary '.fam' module\n" );
   stdOut->writeString( "   -y          write string translation table for the module\n" );

   stdOut->writeString( "\n" );
   stdOut->writeString( "Paths must be in falcon file name format: directory separatros must be slashes [/] and\n" );
   stdOut->writeString( "multiple entries must be entered separed by a semicomma (';')\n" );
   stdOut->writeString( "File names may be set to '-' meaning standard input or output (depending on the option)\n" );
   stdOut->writeString( "\n" );
}

void findModuleName( const String &filename, String &name )
{
   uint32 pos = filename.rfind( "/" );
   if ( pos == csh::npos ) {
      name = filename;
   }
   else {
      name = filename.subString( pos + 1 );
   }

   // find the dot
   pos = name.rfind( "." );
   if ( pos != csh::npos ) {
      name = name.subString( 0, pos );
   }
}


void findModulepath( const String &filename, String &path )
{
   uint32 pos = filename.rfind( "/" );
   if ( pos != csh::npos ) {
      path = filename.subString( 0, pos );
   }
}


void exit_sequence( int exit_value, int errors = 0 )
{

   if ( errors > 0 )
   {
      stdErr = stdErrorStream();
      stdErr->writeString( "falcon: exiting with " );
      String cmpErrors;
      cmpErrors.writeNumber( (int64) errors );
      if( errors > 1 )
         stdErr->writeString( cmpErrors + " errors\n" );
      else
         stdErr->writeString( "1 error\n" );
   }

   if ( options.wait_after )
   {
      stdIn = stdInputStream();
      stdOut = stdOutputStream();
      stdOut->writeString( "Press <ENTER> to terminate\n" );
      Falcon::byte chr;
      stdIn->read( &chr, 1);
   }

   // we must clear the rest with the memalloc functions we had at beginning
   memAlloc = account_alloc;
   memFree = account_free;
   memRealloc = account_realloc;

   options.preloaded.clear();
   options.preloaded.clear();

   exit( exit_value );
}


String get_load_path()
{
   String envpath;
   bool hasEnvPath = Sys::_getEnv( "FALCON_LOAD_PATH", envpath );

   if ( ! hasEnvPath && options.load_path == "" )
      return FALCON_DEFAULT_LOAD_PATH;
   else if ( hasEnvPath ) {
      if ( options.load_path == "" )
         return envpath;
      else
         return options.load_path +";"+ envpath;
   }
   else
      return options.load_path;
}


String get_src_encoding()
{
   if ( options.source_encoding != "" )
      return options.source_encoding;

   if ( options.io_encoding != "" )
      return options.io_encoding;

   String envenc;
   if ( Sys::_getEnv( "FALCON_SRC_ENCODING", envenc ) )
      return envenc;

   if ( Sys::_getEnv( "FALCON_VM_ENCODING", envenc ) )
      return envenc;

   if ( GetSystemEncoding( envenc ) )
      return envenc;

   // we failed.
   return "C";
}


String get_io_encoding()
{
   if ( options.io_encoding != "" )
      return options.io_encoding;

   String ret;
   if ( Sys::_getEnv("FALCON_VM_ENCODING", ret) )
      return ret;

    GetSystemEncoding( ret );
   return ret;
}


int32 readCode( Stream *io, Module *module )
{
   int64 len = io->seekEnd( 0 );
   Falcon::byte *code = (Falcon::byte *) memAlloc( (uint32) len );
   io->seekBegin( 0 );
   io->read( code, (int32) len );
   module->code( code );
   module->codeSize( (uint32) len );

   return (int32)len;
}

Stream *openInputStream( bool bBinary = false )
{
   if( options.input == "" || options.input == "-" )
   {
      return stdIn;
   }

   FileStream *finput = new FileStream;
   finput->open( options.input, FileStream::e_omReadOnly );

   if ( ! finput->good() ) {
      stdErr->writeString( "falcon: can't open input file " + options.input + "\n" );
      delete finput;
      exit_sequence( 1 );
   }

   if ( ! bBinary )
   {
      String ioEncoding = get_src_encoding();
      Transcoder *tcfile = TranscoderFactory( ioEncoding, finput, true );
      if ( tcfile )
      {
         return tcfile;
      }
      else {
         stdOut->writeString( "Fatal: unrecognized encoding '" + ioEncoding + "'.\n\n" );
         exit_sequence( 1 );
      }
   }

   return finput;
}

Stream *openOutputStream( const String &ext, bool bBinary = false )
{
   if( options.output == "-" )
   {
      return stdOut;
   }

   String outName;
   if( options.output == "" )
   {
      if( options.input == "" || options.input == "-" )
         outName = "stdin." + ext;
      else
      {
         String name, path;
         findModuleName( options.input, name );
         findModulepath( options.input, path );
         if ( path != "" )
            outName = path + "/" + name + "." + ext;
         else
            outName = name + "." + ext;
      }
   }
   else {
      outName = options.output;
   }

   FileStream *fout = new FileStream;
   fout->create( outName, FileStream::e_aUserWrite | FileStream::e_aReadOnly );

   if ( ! fout->good() ) {
      stdErr->writeString( "falcon: can't open output file " + outName + "\n" );
      delete fout;
      exit_sequence( 1 );
   }

   if ( ! bBinary )
   {
      String ioEncoding = get_src_encoding();
      Transcoder *tcfile = TranscoderFactory( ioEncoding, fout, true );
      if ( tcfile )
      {
         return tcfile;
      }
      else {
         stdOut->writeString( "Fatal: unrecognized encoding '" + ioEncoding + "'.\n\n" );
         exit_sequence( 1 );
      }
   }

   return fout;
}


void parseOptions( int argc, char **argv, int &script_pos )
{
   bool exitNow = false;

   // option decoding
   for ( int i = 1; i < argc; i++ )
   {
      char *op = argv[i];

      if (op[0] == '-' )
      {
         switch ( op[1] )
         {
            case 'a': options.assemble_only = true; break;
            case 'c': options.compile_only = true; break;
            case 'C': options.check_memory = true; break;
            case 'D':
               if ( op[2] == 0 && i + 1< argc )
                  options.directives.pushBack( new String(argv[++i]) );
               else
                  options.directives.pushBack( new String(op + 2) );
            break;

            case 'e':
               if ( op[2] == 0 && i + 1 < argc ) {
                  options.io_encoding = argv[++i];
               }
               else {
                  options.io_encoding = op + 2;
               }
            break;

            case 'E':
               if ( op[2] == 0 && i + 1< argc ) {
                  options.source_encoding = argv[++i];
               }
               else {
                  options.source_encoding = op + 2;
               }
            break;

            case 'f': options.force_recomp = true; break;
            case 'h': case '?': usage(); exitNow = true;

            case 'L':
               if ( op[2] == 0 && i + 1 < argc )
                  options.load_path = argv[++i];
               else
                  options.load_path = op + 2;
            break;

            case 'l':
               if ( op[2] == 0 && i + 1 < argc )
                  options.module_language = argv[++i];
               else
                  options.module_language = op + 2;
            break;

            case 'm': options.comp_memory = false; break;
            case 'M': options.save_modules = false; break;

            case 'o':
               if ( op[2] == 0 && i + 1< argc )
                  options.output = argv[++i];
               else
                  options.output = op + 2;
            break;

            case 'p':
               if ( op[2] == 0 && i + 1< argc )
                  options.preloaded.pushBack( new String(argv[++i]) );
               else
                  options.preloaded.pushBack( new String(op + 2) );
            break;

            case 'P': options.search_path = true; break;
            case 'r': options.recompile_on_load = false; break;

            case 's': options.via_asm = true; break;
            case 'S': options.assemble_out = true; break;
            case 't': options.tree_out = true; break;
            case 'T': options.parse_ftd = true; break;
            case 'x': options.run_only = true; break;
            case 'v': version(); exitNow = true; break;
            case 'w': options.wait_after = true; break;
            case 'y': options.compile_tltable = true; break;

            default:
               stdErr->writeString( "falcon: unrecognized option '" );
               stdErr->writeString( op );
               stdErr->writeString( "'.\n\n" );
               usage();
               exit_sequence( 1 );
         }
      }
      else {
         options.input = op;
         script_pos = i+1;
         // the other options are for the script.
         break;
      }
   }

   if ( exitNow )
      exit_sequence( 0 );
}


bool apply_directives( Compiler &compiler )
{
   ListElement *dliter = options.directives.begin();
   while( dliter != 0 )
   {
      String &directive = * ((String *) dliter->data());
      // find "="
      uint32 pos = directive.find( "=" );
      if ( pos == String::npos )
      {
         stdErr->writeString( "falcon: directive not in <directive>=<value> syntax'" );
         stdErr->writeString( directive );
         stdErr->writeString( "'\n\n" );
         return false;
      }

      //split the directive
      String dirname( directive, 0, pos );
      String dirvalue( directive, pos + 1 );
      dirname.trim();
      dirvalue.trim();

      // is the value a number?
      int64 number;
      bool result;
      if( dirvalue.parseInt( number ) )
         result = compiler.setDirective( dirname, number );
      else
         result = compiler.setDirective( dirname, dirvalue );

      if ( ! result )
      {
         stdErr->writeString( "falcon: invalid directive or value '" );
         stdErr->writeString( directive );
         stdErr->writeString( "'\n\n" );
         return false;
      }

      dliter = dliter->next();
   }

   return true;
}

//===========================================
// Main Routine
//===========================================

int main( int argc, char *argv[] )
{
   // Function wide statics must be created here, as we may be making memory accounting later on.
   String ioEncoding;
   String sysEncoding;
   if ( ! GetSystemEncoding( sysEncoding ) )
      sysEncoding = "C";
   // TODO: other languages.
   setEngineLanguage( "C" );

   int script_pos=argc;  // presume no script -- stdin.

   // start by checking memory. We'll remove memchecking later if not needed.
   memAlloc = account_alloc;
   memFree = account_free;
   memRealloc = account_realloc;

   EngineData data1;
   Init( data1 );

   // provide minimal i/o during parameter parsing.
   // Todo; first parse parameters and then prepare this.
   stdOut = stdOutputStream();
   stdErr = stdErrorStream();
   stdIn = stdInputStream();

   // Parse options
   parseOptions( argc, argv, script_pos );

   // If memory check is NOT required, reset the default system
   if( ! options.check_memory )
   {
      delete stdOut;
      delete stdErr;
      delete stdIn;

      memAlloc = DflMemAlloc;
      memFree = DflMemFree;
      memRealloc = DflMemRealloc;

      EngineData data1;
      Init( data1 );

      stdOut = stdOutputStream();
      stdErr = stdErrorStream();
      stdIn = stdInputStream();
   }

   // WARNING: the next 3 ops may allocate dynamic string memory.
   // sets default encodings.
   // if I/O encoding is not set, it defaults to system detect;
   if ( options.io_encoding == "" )
      options.io_encoding = sysEncoding;

   //  and if source encoding is not found, it defaults system encoding.
   if ( options.source_encoding == "" )
      options.source_encoding = sysEncoding;

   // now we can get the definitive encodings.
   ioEncoding = get_io_encoding();

   // change stdandard streams to fit needs
   if ( ioEncoding != "" && ioEncoding != "C" && ! options.run_only )
   {
      Transcoder *tcin = TranscoderFactory( ioEncoding, 0, false );
      if ( tcin == 0 )
      {
         stdOut->writeString( "falcon: Fatal: unrecognized encoding '" + ioEncoding + "'.\n\n" );
         exit_sequence( 1 );
      }

      delete stdIn;
      delete stdOut;
      delete stdErr;

      tcin->setUnderlying( new StdInStream, true );
      stdIn = AddSystemEOL( tcin );
      Transcoder *tcout = TranscoderFactory( ioEncoding, new StdOutStream, true );
      stdOut = AddSystemEOL( tcout );
      Transcoder *tcerr = TranscoderFactory( ioEncoding, new StdErrStream, true );
      stdErr = AddSystemEOL( tcerr );
   }

   // our output stream is now completely setup; prepare the error handler
   DefaultErrorHandler *errHand = new DefaultErrorHandler( stdErr );

   // if we have been requested assembly output or tree output, just
   // compile the source and produce the output.
   if ( options.assemble_out || options.tree_out )
   {
      Module *module = new Module();
      Stream *input = openInputStream();

      Compiler compiler( module, input );
      // apply required directives
      if ( ! apply_directives( compiler ) )
      {
         exit_sequence(1);
      }

      compiler.errorHandler( errHand );

      // is input an FTD?
      if ( options.parse_ftd ||
           options.input.rfind( ".ftd" ) == options.input.length() - 4 )
      {
         compiler.parsingFtd( true );
      }

      if( ! compiler.compile() )
      {
         if ( input != stdIn )
            delete input;

         exit_sequence( 1, compiler.errors() );
      }

      Stream *out;
      if ( options.assemble_out )
      {
         out = openOutputStream( "fas" );
         GenHAsm hasm( out );
         hasm.generatePrologue( compiler.module() );
         hasm.generate( compiler.sourceTree() );
      }
      else
      {
         out = openOutputStream( "ftr" );
         GenTree tree( out );
         ((Generator *) &tree)->generate( compiler.sourceTree() );
      }

      if ( input != stdIn )
         delete input;

      if ( out != stdOut )
         delete out;

      module->decref();
      exit_sequence( 0 );
   }

   // ======================================
   // Load and execute the module
   //

   // 1. Ready the module loader
   FlcLoader *modLoader = new FlcLoader( get_load_path() );

   // set the module preferred language; ok also if default ("") is used
   modLoader->setLanguage( options.module_language );

   if ( ! apply_directives( modLoader->compiler() ) )
   {
      exit_sequence(1);
   }

   if( options.input != "" && options.input != "-" )
   {
      String source_path;
      findModulepath( options.input, source_path );
      if( source_path != "" )
      {
         modLoader->addSearchPath( source_path );
      }
   }

   modLoader->errorHandler( errHand );
   modLoader->compileInMemory( options.comp_memory );
   modLoader->compileViaAssembly( options.via_asm );

   // save the main module also if compile only option is set
   modLoader->saveModules( options.save_modules || options.compile_only );

   //... but disable if compiling tltables.
   if ( options.compile_tltable )
      modLoader->saveModules( false );

   modLoader->alwaysRecomp( options.compile_only || options.force_recomp );
   modLoader->sourceEncoding( get_src_encoding() );
   // normally, save is not mandatory, unless we compile them our own
   // should be the default, but we reset it.
   modLoader->saveMandatory( false );

   // should we forcefully consider input as ftd?
   modLoader->compileTemplate( options.parse_ftd );

   // If we have to assemble or compile just a module...
   if ( options.assemble_only || options.compile_only )
   {
      if( options.assemble_only )
         modLoader->sourceIsAssembly( true );

      // force not to save modules, we're saving it on our own
      modLoader->saveModules( false );

      Module *mod;
      if ( options.input == "" || options.input == "-" )
         mod = modLoader->loadSource( stdIn, "<stdin>" );
      else
         mod = modLoader->loadSource( options.input );

      // if we had a module, save it at the right place.
      if ( mod != 0 )
      {
         // should we save the module ?
         if ( options.assemble_only || options.compile_only )
         {
            Stream *modstream = openOutputStream( "fam" );
            mod->save( modstream );
            modstream->close();
         }
      }

      // whatever happened, we go to exit now.
      exit_sequence( modLoader->compileErrors() > 0 , modLoader->compileErrors() );
   }

   // At this point, we can ignore sources if only willing to run
   // with ignore source option, the flcLoader will bypass loadSource and use loadModule.
   modLoader->ignoreSources( options.run_only );

   // Load the modules
   Module *mainMod;

   // the module loader has been already configured to to the job.
   if( options.input != "" && options.input != "-" )
   {
      mainMod = modLoader->loadFile( options.input, FlcLoader::t_defaultSource );
   }
   else {
      if( options.run_only )
         mainMod = modLoader->loadModule( stdIn );
      else
         mainMod = modLoader->loadSource( stdIn, "<stdin>" );
   }

   if ( mainMod == 0 )
   {
      exit_sequence( 1, modLoader->compileErrors() );
   }

   // should we just write the string table?
   if( options.compile_tltable )
   {
      if( mainMod->stringTable().internatCount() > 0 )
      {
         Stream *tplstream = openOutputStream( "temp.ftt" );
         if( tplstream )
         {
            // Wrap using default encoding
            if ( options.source_encoding != "" && options.source_encoding != "C" )
               tplstream = TranscoderFactory( options.source_encoding, tplstream, false );
            mainMod->saveTableTemplate( tplstream, options.source_encoding );
            tplstream->close();
         }
         else
            exit_sequence( 1, 0 );
      }

      exit_sequence( 0 , 0 );
   }

   // Create the runtime using the given module loader.
   Runtime *runtime = new Runtime( modLoader );

   // now that we have the main module, inject other requested modules
   ListElement *pliter = options.preloaded.begin();
   while( pliter != 0 )
   {
      Module *module = modLoader->loadName( * ((String *) pliter->data()) );
      if ( ! module )
         exit_sequence( 1 );
      if ( ! runtime->addModule( module ) )
         exit_sequence( 1 );

      // abandon our reference to the injected module
      module->decref();

      pliter = pliter->next();
   }

   // then add the main module
   if ( ! runtime->addModule( mainMod ) )
   {
      // addmodule should already have raised the error.
      exit_sequence( 1 );
   }

   // abandon our reference to the main module
   mainMod->decref();

   //===========================================
   // Prepare the virtual machine
   //
   VMachine *vmachine = new VMachine;

   //redirect the VM streams to ours.
   // The machine takes ownership of the streams, so they won't be useable anymore
   // after the machine destruction.
   vmachine->stdIn( stdIn );
   vmachine->stdOut( stdOut );
   vmachine->stdErr( stdErr );
   // I have given real process streams to the vm
   vmachine->hasProcessStreams( true );

   // Set the error handler
   vmachine->errorHandler( errHand );

   // push the core module
   Module *core = core_module_init();
   LiveModule *res = vmachine->link( core );
   fassert( res != 0 );  // should not fail
   core->decref();

   // prepare environment
   Item *item_args = vmachine->findGlobalItem( "args" );
   fassert( item_args != 0 );
   CoreArray *args = new CoreArray( vmachine, argc - script_pos );

   for ( int ap = script_pos; ap < argc; ap ++ ) {
      String *cs = new GarbageString( vmachine );
      if ( ! TranscodeFromString( argv[ap], ioEncoding, *cs ) )
      {
         cs->bufferize( argv[ap] );
      }

      args->append( cs );
   }

   item_args->setArray( args );

   Item *script_name = vmachine->findGlobalItem( "scriptName" );
   fassert( script_name != 0 );
   *script_name = new GarbageString( vmachine, mainMod->name() );

   Item *script_path = vmachine->findGlobalItem( "scriptPath" );
   fassert( script_path != 0 );
   *script_path = new GarbageString( vmachine, mainMod->path() );

   // Link the runtime in the VM.
   if( ! vmachine->link( runtime ) )
   {
      delete vmachine;
      delete runtime;

      // a failed link means undefined symbols or error in object init.
      exit_sequence( 1 );
   }

   // now the machine can be launched.
   vmachine->launch();

   // manage suspension events.
   while( vmachine->lastEvent() == VMachine::eventSuspend ) {
      stdOut->writeString( "Virtual machine suspended. Please enter an event:\n" );
      String ret;
      uint32 chr;
      while( stdIn->get( chr ) && chr != '\n' )
         if ( chr != '\r' )
            ret.append( chr );

      vmachine->resume( new GarbageString( vmachine, ret ) );
      // items in resume are not automatically stored in the GC, so we can destroy
      // the string here.
   }

   bool exitSeq = vmachine->regA().isInteger();
   int32 exitVal;

   if ( exitSeq )
      exitVal = (int32) vmachine->regA().asInteger();
   else
      exitVal = 0;

   delete vmachine;
   delete runtime;
   delete modLoader;
   delete errHand;

   // to de-account memory
   options.io_encoding = "";
   options.source_encoding = "";
   ioEncoding = "";

   if( options.check_memory )
   {
      // take memory now (should be 0 after re-creating the streams).
      long mem = s_allocatedMem;
      long blocks = s_outBlocks;

      // recreate an output stream
      stdOut = stdOutputStream();

      String temp = " Allocated Memory / Allocated Blocks : ";
      temp.writeNumber( (int64) mem );
      temp += " / ";
      temp.writeNumber( (int64) blocks );
      stdOut->writeString( "-------------------------------------------------------\n"
                           "Memory report:\n" );
      stdOut->writeString( temp );
      stdOut->writeString( s_validAlloc == 1 ? "  (valid blocks)" : "  (some deallocation error)" );
      stdOut->writeString( "\n-------------------------------------------------------\n" );

      delete stdOut;
   }

   exit_sequence( exitSeq ? exitVal : 0 );
   return 0; // to make the compiler happy
}

/* end of falcon.cpp */
