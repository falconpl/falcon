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

#include "mxml_fm.h"

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
   Falcon::Module *self = new Falcon::Feathers::ModuleMXML();
   return self;
}

#endif

/* end of sdl.cpp */

