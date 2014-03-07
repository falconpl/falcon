/*
   FALCON - The Falcon Programming Language.
   FILE: options.h

   Options storage for falcon compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 26 Jul 2011 10:31:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Options storage for falcon compiler.
*/

#ifndef _FALCON_OPTIONS_H_
#define _FALCON_OPTIONS_H_

#include <falcon/string.h>

#include <list>
#include <map>

namespace Falcon {

/** Options storage for falcon compiler.
 
   This class is just a nice place to store options for the compiler and their defaults.
*/

class FalconOptions
{
public:
   typedef std::list<String> StringList;
   typedef std::map<String, String> StringMap;
   
   String input;
   String output;
   String load_path;
   String io_encoding;
   String source_encoding;
   String module_language;
#ifndef NDEBUG
   String trace_file;
#endif

   StringMap directives;
   StringMap defines;
   StringList preloaded;
   
   String test_dir;
   String log_file;

   bool compile_only;
   bool run_only;
   bool tree_out;
   
   bool search_path;
   bool force_recomp;
   bool check_memory;

   bool ignore_sources;
   bool save_modules;
   bool wait_after;
   bool parse_ftd;

   bool interactive;

   bool ignore_syspath;

   bool errOnStdout;
   bool testMode;
   String test_category;
   String test_prefix;
   bool list_tests;
   int log_level;

   int num_processors;
   int print_grammar;

   bool m_bEval;
   String m_sEval;
   int m_errorReportLevel;

   FalconOptions();

   void parse( int argc, char **argv, int &script_pos );
   void usage( bool deep = false );
   void version();

   /** Returns true if the parsed options required an immediate exit. */
   bool wasJustInfo() const { return m_justinfo; }

   void getErrorReportMode( bool& bAddPath, bool& bAddParams, bool& bAddSign ) const;
public:
   void modalGiven();
   void parseDirective( const String& str );
   void parseDefine( const String& str );   
   void parseEqString( const String& str, String& key, String& value );
   bool m_modal;
   bool m_justinfo;
};

} // namespace Falcon

#endif

/* end of options.h */
