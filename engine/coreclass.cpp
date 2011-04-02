/*
   FALCON - The Falcon Programming Language.
   FILE: coreclass.cpp

   Core Class implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu Jan 20 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   CoreClass implementation
*/

#include <falcon/cclass.h>
#include <falcon/coreobject.h>
#include <falcon/vm.h>
#include <falcon/itemdict.h>

namespace Falcon {

CoreClass::CoreClass()
{
}

bool CoreClass::derivedFrom( const String &className ) const
{

   return false;
}


CoreClass::~CoreClass()
{
}

}


/* end of coreclass.cpp */
