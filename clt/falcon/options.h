/*
   FALCON - The Falcon Programming Language.
   FILE: options.h

   Options storage for falcon compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven set 10 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Options storage for falcon compiler.
*/

#ifndef flc_options_H
#define flc_options_H

#include <falcon/string.h>
#include <falcon/genericlist.h>

/** Options storage for falcon compiler
   This class is just a nice place to store options for the compiler and their defaults.
*/

namespace Falcon {

class HOptions
{
public:

   String input;
   String output;
   String load_path;
   String io_encoding;
   String source_encoding;
   String module_language;
   List preloaded;
   List directives;
   List defines;

   bool compile_only;
   bool run_only;
   bool tree_out;
   bool assemble_out;
   bool search_path;
   bool force_recomp;
   bool check_memory;

   bool comp_memory;
   bool recompile_on_load;
   bool save_modules;
   bool wait_after;
   bool parse_ftd;

   bool compile_tltable;
   bool interactive;

   HOptions():
      input( "" ),
      output( "" ),
      load_path( "" ),
      io_encoding( "" ),
      source_encoding( "" ),
      module_language( "" ),
   
      compile_only( false ),
      run_only( false ),
      tree_out( false ),
      assemble_out( false ),
      search_path( false ),
      force_recomp( false ),
      check_memory( false ),

      comp_memory( true ),
      recompile_on_load( true ),
      save_modules( true ),
      wait_after( false ),
      parse_ftd( false ),
      
      compile_tltable( false ),
      interactive( false )
   {}
};

}

#endif

/* end of options.h */
