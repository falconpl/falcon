/*
   FALCON - The Falcon Programming Language.
   FILE: mxml.cpp

   The minimal XML support module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 01 Mar 2008 10:23:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The mxml module - main file.
*/

#include <falcon/module.h>
#include "version.h"

#include "mxml.h"
#include "mxml_ext.h"

FALCON_MODULE_DECL
{
   Falcon::Module *self = new Falcon::Ext::MXMLModule();
   return self;
}

/* end of sdl.cpp */

