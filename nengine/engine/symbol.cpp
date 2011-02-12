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

namespace Falcon {


Symbol::Symbol( type_t t, const String& name ):
      m_type(t),
      m_name( name )
{}

Symbol::Symbol( const Symbol& other ):
   m_type( other.m_type ),
   m_name( other.m_name )
{}


Symbol::~Symbol()
{
}

}

/* end of symbol.cpp */
