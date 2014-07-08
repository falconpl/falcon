/*
   FALCON - The Falcon Programming Language.
   FILE: confparser.cpp

   Configuration parser module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 08 Aug 2013 18:28:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/confparser.cpp"
#include "confparser_fm.h"

FALCON_MODULE_DECL
{
   // setup DLL engine common data

   Falcon::Module *self = new Falcon::Feathers::ModuleConfparser();
   return self;
}

/* end of confparser.cpp */


