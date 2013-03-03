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

#include <falcon/module.h>
#include <falcon/closure.h>

namespace Falcon {

Symbol::Symbol():
         m_counter(1)
{
   // leave all unconfigured.
}

Symbol::Symbol( const String& name ):
   m_name(name),
   m_counter(1)
{
}
   

   
Symbol::Symbol( const Symbol& other ):
   m_name(other.m_name),
   m_counter(1)
{
}


Symbol::~Symbol()
{
}


void Symbol::incref()
{
   Engine::refSymbol( this );
}

void Symbol::decref()
{
   Engine::releaseSymbol( this );
}



}

/* end of symbol.cpp */
