/*
   FALCON - The Falcon Programming Language.
   FILE: exprsym.cpp

   Syntactic tree item definitions -- expression elements -- symbol.

   Pure virtual class base for the various symbol types.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/symbol.h>
#include <falcon/exprsym.h>

namespace Falcon {

ExprSymbol::ExprSymbol( const ExprSymbol& other ):
      Expression( other ),
      m_symbol(other.m_symbol),
      m_lvalue(false)
{
   // apply is created by symbols
   // TODO raise error in debug if apply is not present.
}

ExprSymbol::ExprSymbol( Symbol* target ):
   Expression( t_symbol ),
   m_symbol( target ),
   m_lvalue(false)
{
   // apply is created by symbols
   // TODO raise error in debug if apply is not present.
}


ExprSymbol::~ExprSymbol()
{
   // nothig to do
}

void ExprSymbol::describe( String& val ) const
{
   val = m_symbol->type() == Symbol::t_unknown_symbol ? 
      "/* unknown */" + m_symbol->name() :
      m_symbol->name();
}


void ExprSymbol::serialize( DataWriter* ) const
{
   // TODO
}


void ExprSymbol::deserialize( DataReader* )
{
   // TODO
}

}

/* end of exprsym.cpp */
