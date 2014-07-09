/*
   FALCON - The Falcon Programming Language.
   FILE: regex_fm.cpp

   Regular expression extension
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/regex/regex.cpp"

#undef PCRE_EXP_DATA_DECL
// create the data function pointers in this code
#define PCRE_EXP_DATA_DECL

#include "pcre.h"

#include "regex_ext.h"
#include "regex_fm.h"
#include "regex_ext.h"

#include <stdio.h>

namespace Falcon {
namespace Feathers {

ModuleRegex::ModuleRegex():
   Module(FALCON_FEATHER_REGEX_NAME)
{
   // initialize PCRE

   pcre_malloc = malloc;
   pcre_free = free;
   pcre_stack_malloc = malloc;
   pcre_stack_free = free;

   addMantra(new Falcon::Ext::ClassRegex);

   //==================================================
   // Error class

   addMantra(new Falcon::Ext::ClassRegexError );
}

ModuleRegex::~ModuleRegex()
{};

}
}
/* end of regex_fm.cpp */
