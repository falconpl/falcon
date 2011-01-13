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


Symbol::Symbol( operator_t type, const String& name ):
      Expression( type ),
      m_name( name ),
      m_lvalue( false )
{}

Symbol::Symbol( const Symbol& other ):
   Expression(other),
   m_name( other.m_name ),
   m_lvalue( other.m_lvalue )
{}


Symbol::~Symbol()
{
}

void Symbol::serialize( Stream* s ) const
{
   // TODO
}

void Symbol::toString( String& val ) const
{
   val = "&" + m_name;
}

void Symbol::deserialize( Stream* s )
{
   // TODO
}

}

/* end of symbol.cpp */
