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
#include <falcon/pcode.h>

namespace Falcon {

ExprSymbol::ExprSymbol( const ExprSymbol& other ):
      Expression( other ),
      m_pslv(this),
      m_symbol(other.m_symbol)
{
   apply = other.apply;
   m_pstep_lvalue = &m_pslv;
   m_pstep_lvalue->apply = other.m_pstep_lvalue->apply;
}

ExprSymbol::ExprSymbol( Symbol* target ):
   Expression( t_symbol ),
   m_pslv(this),
   m_symbol( target )
{
   m_pstep_lvalue = &m_pslv;
}


void ExprSymbol::precompileLvalue( PCode* pcode ) const
{
   pcode->pushStep( m_pstep_lvalue );
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


void ExprSymbol::PStepLValue::describe( String& s ) const
{
   m_owner->describe( s );
}

}

/* end of exprsym.cpp */
