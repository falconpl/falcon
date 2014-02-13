/*
   FALCON - The Falcon Programming Language.
   FILE: sybmol.cpp

   Syntactic tree item definitions -- expression elements -- symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/symbol.h>
#include <falcon/stream.h>
#include <falcon/vm.h>
#include <falcon/callframe.h>
#include <falcon/function.h>
#include <falcon/module.h>
#include <falcon/itemarray.h>
#include <falcon/fassert.h>
#include <falcon/modspace.h>
#include <falcon/stdhandlers.h>

#include <falcon/module.h>
#include <falcon/closure.h>

namespace Falcon {

void Symbol::incref() const
{
   Engine::refSymbol( this );
}

void Symbol::decref() const
{
   Engine::releaseSymbol( this );
}

Class* Symbol::handler()
{
   static Class* h = Engine::instance()->handlers()->symbolClass();
   return h;
}

}

/* end of symbol.cpp */
