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

#include "options.h"
#include <iostream>

using namespace std;
using namespace Falcon;


FalconOptions::FalconOptions():
   input( "" ),
   output( "" ),
   load_path( "" ),
   io_encoding( "" ),
   source_encoding( "" ),
   module_language( "" ),

   compile_only( false ),
   run_only( false ),
   tree_out( false ),
   search_path( false ),
   force_recomp( false ),
   check_memory( false ),

   comp_memory( true ),
   recompile_on_load( true ),
   save_modules( true ),
   wait_after( false ),
   parse_ftd( false ),

   compile_tltable( false ),
   interactive( false ),
   ignore_syspath( false ),
   errOnStdout(false),
   m_modal( false ),
   m_justinfo( false )
   
{}


void FalconOptions::usage( bool deep )
{
   cout
      << "Usage:\n" << endl
      << "       falcon (-c|-S|-t) [c_opts] [-o<output>] module" << endl
      << "       falcon -y [-o<output>] module" << endl
      << "       falcon -x [c_options] module" << endl
      << "       falcon [c_opts] [r_opts] module [script options]" << endl
      << "       falcon -i [-p module] ... [-p module]" << endl
      << endl;

   if( deep )
   {
      cout
      << "Modal options:" << endl
      << "   -c          compile the given source(s) in a .fam module" << endl
      << "   -i          interactive mode" << endl
      << "   -t          generate a syntactic tree (for logic debug)" << endl
      << "   -x          execute a binary '.fam' module" << endl
      << "   -y          write string translation table for the module" << endl
      << "   --cgi       execute in GGI mode" << endl
      << endl
      << "Compilation options (c_opts):" << endl
      << "   -d          Set directive (as <directive>=<value>)" << endl
      << "   -D          Set constant (as <constant>=<value>)" << endl
      << "   -E <enc>    Source files are in <enc> encoding (overrides -e)" << endl
      << "   -f          force recompilation of modules even when .fam are found" << endl
      << "   -m          do NOT compile in memory (use temporary files)" << endl
      << "   -T          consider given [module] as .ftd (template document)" << endl
      << endl
      << "Run options (r_opts):" << endl
      << "   -C          check for memory allocation correctness" << endl
      << "   -e <enc>    set given encoding as default for VM I/O" << endl
#ifndef NDEBUG
      << "   -F <file>   Output TRACE debug statements to <file>" << endl
#endif
      << "   -l <lang>   Set preferential language of loaded modules" << endl
      << "   -L <path>   Add path for 'load' directive (start with ';' remove std paths)" << endl
      << "   -M          do NOT save the compiled modules in '.fam' files" << endl
      << "   -p <module> preload (pump in) given module" << endl
      << "   -P          ignore system PATH (and FALCON_LOAD_PATH envvar)" << endl
      << "   -r          do NOT recompile sources (ignore sources)" << endl
      << endl
      << "General options:" << endl
      << "   -h/-?       display usage" << endl
      << "   -H          this help" << endl
      << "   -o <fn>     output to <fn> instead of [module.xxx]" << endl
      << "   -s          Send error description to stdOut instead of stdErr" << endl
      << "   -v          print copyright notice and version and exit" << endl
      << "   -w          add an extra console wait after program exit" << endl
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

            case 'm': comp_memory = false; break;
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
            case 'r': recompile_on_load = false; break;

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

            // else just fallthrough

            default:
               cerr << "falcon: unrecognized option '" << op << "'."<< endl << endl;
               usage(false);
               m_justinfo = true;
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
   cout << "Version "  << FALCON_VERSION " (" FALCON_VERSION_NAME ")" << endl << endl;
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
   length_t pos = str.find( "=" );
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

/* options.cpp */
