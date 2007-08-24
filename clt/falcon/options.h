/*
   FALCON - The Falcon Programming Language.
   FILE: options.h
   $Id: options.h,v 1.4 2007/08/02 23:41:39 jonnymind Exp $

   Options storage for falcon compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven set 10 2004
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
   List preloaded;

   bool compile_only;
   bool assemble_only;
   bool run_only;
   bool tree_out;
   bool assemble_out;
   bool search_path;
   bool force_recomp;
   bool check_memory;

   bool via_asm;
   bool comp_memory;
   bool recompile_on_load;

   bool save_modules;

   bool wait_after;

   HOptions():
      input( "" ),
      output( "" ),
      load_path( "" ),
      compile_only( false ),
      check_memory( false ),
      assemble_only( false ),
      assemble_out( false ),
      run_only( false ),
      tree_out( false ),
      via_asm( false ),
      search_path( false ),
      recompile_on_load( true ),
      comp_memory( true ),
      save_modules( true ),
      force_recomp( false ),
      io_encoding( "" ),
      source_encoding( "" ),
      wait_after( false )
   {}
};

}

#endif

/* end of options.h */
