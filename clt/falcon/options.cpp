/*
   FALCON - The Falcon Programming Language.
   FILE: options.cpp

   Falcon compiler and interpreter - options file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 26 Jul 2011 10:31:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon compiler and interpreter - options file
*/

#include <iostream>

#include "options.h"

using namespace std;

#include <falcon/textwriter.h>
#include <falcon/stdstreams.h>
#include <falcon/sp/sourceparser.h>

namespace Falcon {
FalconOptions::FalconOptions():
   input( "" ),
   output( "" ),
   load_path( "" ),
   io_encoding( "C" ),
   source_encoding( "C" ),
   module_language( "" ),
   log_file("%"),

   compile_only( false ),
   run_only( false ),
   tree_out( false ),
   search_path( false ),
   force_recomp( false ),
   check_memory( false ),
   
   ignore_sources( false ),
   save_modules( true ),
   wait_after( false ),
   parse_ftd( false ),

   compile_tltable( false ),
   interactive( false ),
   ignore_syspath( false ),
   errOnStdout(false),
   testMode(false),
   list_tests(false),
   log_level(-1),
   num_processors(0),
   m_bEval( false ),

   m_modal( false ),
   m_justinfo( false )
{}


void FalconOptions::usage( bool deep )
{
   cout
      << "Usage:\n" << endl
      << "       falcon (-c|-S|-t) [c_opts] [-o<output>] module" << endl
      << "       falcon [-p <mod> -p <mod>...] -r \"program to be evaluated\"" << endl
      << "       falcon -y [-o<output>] module" << endl
      << "       falcon -x [c_options] module" << endl
      << "       falcon --test <directory> [test_opts]" << endl
      << "       falcon [c_opts] [r_opts] module [script options]" << endl
      << "       falcon -i [-p module] ... [-p module]" << endl
      << endl;

   if( deep )
   {
      cout
      << "Modal options:" << endl
      << "  -c           compile the given source(s) in a .fam module" << endl
      << "  -i           interactive mode" << endl
      << "  -t           generate a syntactic tree (for logic debug)" << endl
      << "  -x           execute a binary '.fam' module" << endl
      << "  -y           write string translation table for the module" << endl
      << "  -r <text>    Run (evaluate) given text and exit" << endl
      << "  --cgi        execute in GGI mode" << endl
      << "  --grammar    Prints the language grammar definition and exits" << endl
      << "  --test <dir> Execute tests in the given directory" << endl
      << endl
      << "Compilation options (c_opts):" << endl
      << "  -d           Set directive (as <directive>=<value>)" << endl
      << "  -D           Set constant (as <constant>=<value>)" << endl
      << "  -E <enc>     Source files are in <enc> encoding (overrides -e)" << endl
      << "  -f           force recompilation of modules even when .fam are found" << endl
      << "  -T           consider given [module] as .ftd (template document)" << endl
      << endl
      << "Test mode options (test_opts):" << endl
      << "  --cat <cat>   Test this category only" << endl
      << "  --tlist       Just list the test descriptions" << endl
      << "  --tpref <prf> Just run the tests with the given prefix" << endl
      << endl
      << "Run options (r_opts):" << endl
      << "  -C           check for memory allocation correctness" << endl
      << "  -e <enc>     set given encoding as default for VM I/O" << endl
#ifndef NDEBUG
      << "  -F <file>    Output TRACE debug statements to <file> (local platform file format)" << endl
#endif
      << "  -l <lang>    Set preferential language of loaded modules" << endl
      << "  -L <path>    Add path for 'load' and 'import' directives" << endl
      << "  -M           do NOT save the compiled modules in '.fam' files" << endl
      << "  -p <module>  preload (pump in) given module" << endl
      << "  -P           ignore system PATH (and FALCON_LOAD_PATH envvar)" << endl
      << "  -I           Ignore sources" << endl
      << "  --ll <lvl>   Set system log level 0: critical, 7: debug, -1: off"  << endl
      << "  --log <file> Send System log to file (-:stdout, %:stderr)"  << endl
      << "  --prc <num>  Set number of VM processors (0 = match CPU count)" << endl
      << endl
      << "General options:" << endl
      << "  -h/-?        display usage" << endl
      << "  -H           this help" << endl
      << "  -o <fn>      output to <fn> instead of [module.xxx]" << endl
      << "  -s           Send error description to stdOut instead of stdErr" << endl
      << "  -v           print copyright notice and version and exit" << endl
      << "  -w           add an extra console wait after program exit" << endl
      << "" << endl
      << "Paths must be in falcon file name format: directory separators must be slashes [/] and" << endl
      << "multiple entries must be entered separed by a semicolon (';')" << endl
      << "File names may be set to '-' meaning standard input or output (depending on the option)" << endl
      << endl;
   }
   else
   {
      cout
      << "Use '-H' option to get more help." << endl;
   }
}


void FalconOptions::modalGiven()
{
   if (m_modal)
   {
      throw String("multiple modal options selected");
   }

   m_modal = true;
}


void FalconOptions::parse( int argc, char **argv, int &script_pos )
{
   // option decoding
   for ( int i = 1; i < argc; i++ )
   {
      char *op = argv[i];

      if ( op[0] == '-' )
      {
         switch ( op[1] )
         {
            case 'c': modalGiven(); compile_only = true; break;
            case 'C': check_memory = true; break;
            case 'd':
               if ( op[2] == 0 && i + 1< argc )
                  parseDirective( argv[++i] );
               else
                  parseDirective( op + 2 );
               break;

            case 'D':
               if ( op[2] == 0 && i + 1< argc )
                  parseDefine( argv[++i] );
               else
                  parseDefine( op+2 );
               break;

            case 'e':
               if ( op[2] == 0 && i + 1 < argc )
               {
                  io_encoding = argv[++i];
               }
               else
               {
                  io_encoding = op + 2;
               }
               break;

            case 'r':
               modalGiven();
               m_bEval = true;
               if ( op[2] == 0 && i + 1 < argc )
               {
                  m_sEval = argv[++i];
               }
               else
               {
                  m_sEval = op + 2;
               }
               break;

            case 'E':
               if ( op[2] == 0 && i + 1< argc )
               {
                  source_encoding = argv[++i];
               }
               else
               {
                  source_encoding = op + 2;
               }
               break;

            case 'f': force_recomp = true; break;
#ifndef NDEBUG
            case 'F':
               if ( op[2] == 0 && i + 1 < argc )
                  trace_file = argv[++i];
               else
                  trace_file = op + 2;
               break;
#endif
            case 'h': case '?': usage(false); m_justinfo = true; break;
            case 'H': usage(true); m_justinfo = true; break;
            case 'i': modalGiven(); interactive = true; break;

            case 'L':
               if ( op[2] == 0 && i + 1 < argc )
                  load_path = argv[++i];
               else
                  load_path = op + 2;
               break;

            case 'l':
               if ( op[2] == 0 && i + 1 < argc )
                  module_language = argv[++i];
               else
                  module_language = op + 2;
               break;

            case 'M': save_modules = false; break;

            case 'o':
               if ( op[2] == 0 && i + 1< argc )
                  output = argv[++i];
               else
                  output = op + 2;
               break;

            case 'p':
               if ( op[2] == 0 && i + 1< argc )
                  preloaded.push_back( argv[++i] );
               else
                  preloaded.push_back( op + 2 );
               break;

            case 'P': ignore_syspath = true; break;
            case 'I': ignore_sources = true; break;

            case 's': errOnStdout = true; break;
            case 't': modalGiven(); tree_out = true; break;
            case 'T': parse_ftd = true; break;
            case 'x': run_only = true; break;
            case 'v': version(); m_justinfo = true; break;
            case 'w': wait_after = true; break;
            case 'y': modalGiven(); compile_tltable = true; break;

            case '-':
               if( String( op+2 ) == "cgi" )
               {
                  errOnStdout = true;
                  preloaded.push_back( "cgi" );
                  break;
               }
               else if( String( op+2 ) == "test" )
               {
                  testMode = true;
                  test_dir = String( argv[++i] );
                  break;
               }
               else if (String( op+2 ) == "cat" )
               {
                  test_category = String( argv[++i] );
                  break;
               }
               else if (String( op+2 ) == "tpref" )
               {
                  test_prefix = String( argv[++i] );
                  break;
               }
               else if (String( op+2 ) == "tlist" )
               {
                  list_tests = true;
                  break;
               }
               else if( String( op+2 ) == "ll" )
               {
                  log_level = atoi( argv[++i] );
                  break;
               }
               else if( String( op+2 ) == "log" )
               {
                  log_file = String( argv[++i] );
                  break;
               }
               else if( String( op+2 ) == "prc" )
               {
                  num_processors = atoi( argv[++i] );
                  break;
               }
               else if( String( op+2 ) == "grammar" )
               {
                  StdOutStream sout;
                  TextWriter tw(&sout);
                  SourceParser p;
                  p.MainProgram.render(tw);

                  m_justinfo = true;
                  break;
               }
               /* no break */

            default:
               cerr << "falcon: unrecognized option '" << op << "'."<< endl << endl;
               usage(false);
               m_justinfo = true;
               break;
         }
      }
      else
      {
         input = op;
         script_pos = i+1;
         // the other m_options are for the script.
         break;
      }
   }
}


void FalconOptions::version()
{
   cout << "Falcon compiler and interpreter." << endl;
   cout << "Version "  << FALCON_VERSION << " (" FALCON_VERSION_NAME ")" << endl << endl;
}


void FalconOptions::parseDirective( const String& str )
{
   String key;
   String value;
   
   parseEqString(str, key, value);
   if( value == "" )
   {
      cerr << "falcon: invalid directive value '" << str.c_ize() << "'."<< endl << endl;
      m_justinfo = true;
      return;
   }
   
   // let the parsing of the directive content to the compiler later on.
   directives[key] = value;
}


void FalconOptions::parseDefine( const String& str )
{
   String key;
   String value;
   
   parseEqString(str, key, value);
   // There's no invalid values for defines, but empty key is not allowed.
   if( key == "" )
   {
      cerr << "falcon: invalid define value '" << str.c_ize() << "'."<< endl << endl;
      m_justinfo = true;
      return;
   }
   
   defines[key] = value;
}


void FalconOptions::parseEqString( const String& str, String& key, String& value )
{
   length_t pos = str.find( '=' );
   if( pos != String::npos )
   {
      key = str.subString(0, pos);
      if( pos < str.length() )
      {
        value = str.subString(pos+1);
      }
      else
      {
         value.size(0);
      }
   }
   else
   {
      key = str;
      value.size(0);
   }
   
   key.bufferize();
   value.bufferize();
}
}

/* options.cpp */
