/*
   FALCON - The Falcon Programming Language.
   FILE: cmdline.h

   Command line parser module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 08 Aug 2013 18:28:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "module_cmdline.h"

FALCON_MODULE_DECL
{
   // setup DLL engine common data

   ::Falcon::Module *self = new ::Falcon::Feathers::ModuleCmdline;

   return self;
}
