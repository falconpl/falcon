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

#undef SRC
#define SRC "engine/psteps/exprsym.cpp"

#include <falcon/symbol.h>
#include <falcon/pcode.h>
#include <falcon/vmcontext.h>

#include <falcon/psteps/exprsym.h>

namespace Falcon {

ExprSymbol::ExprSymbol( const ExprSymbol& other ):
      Expression( other ),
      m_symbol(other.m_symbol),
      m_pslv(this)
{
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
   m_pstep_lvalue->apply = other.m_pstep_lvalue->apply;
}

ExprSymbol::ExprSymbol( Symbol* target ):
   Expression( t_symbol ),
   m_symbol( target ),
   m_pslv(this)
{
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
}

ExprSymbol::ExprSymbol( const String& name ):
   Expression( t_symbol ),
   m_name(name),
   m_symbol( 0 ),
   m_pslv(this)
{
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
}


void ExprSymbol::precompileLvalue( PCode* pcode ) const
{
   pcode->pushStep( m_pstep_lvalue );
}

void ExprSymbol::precompileAutoLvalue( PCode* pcode, const PStep* activity, bool, bool bSave ) const
{
   // We do this, but the parser should have blocked us...
   precompile( pcode );             // access -- prepare params
   
   // eventually save the value
   if( bSave )
   {
      pcode->pushStep( &m_pstepSave );
   }
   
   pcode->pushStep( activity );     // action
   
   // no restore

   pcode->pushStep( m_pstep_lvalue );
   if( bSave )
   {     
      pcode->pushStep( &m_pstepRemove );
   }  
}

ExprSymbol::~ExprSymbol()
{
   // nothig to do
}

const String& ExprSymbol::name() const
{
   if ( m_symbol != 0 )
   {
      return m_symbol->name();
   }
   return m_name;
}

void ExprSymbol::describeTo( String& val ) const
{
   val = m_symbol != 0 ? m_symbol->name() : m_name;
}


void ExprSymbol::serialize( DataWriter* ) const
{
   // TODO
}


void ExprSymbol::deserialize( DataReader* )
{
   // TODO
}


void ExprSymbol::PStepLValue::describeTo( String& s ) const
{
   m_owner->describeTo( s );
}



void ExprSymbol::PStepSave::describeTo( String& s ) const
{
   s = "Symbol internal save";
}


void ExprSymbol::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol* es = static_cast<const ExprSymbol*>(ps);
   ctx->pushData( *es->m_symbol->value(ctx) );
}


void ExprSymbol::PStepLValue::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol::PStepLValue* es = static_cast<const ExprSymbol::PStepLValue*>(ps);
   *es->m_owner->m_symbol->value(ctx) = ctx->topData();
}


void ExprSymbol::PStepSave::apply_( const PStep*, VMContext* ctx )
{
   ctx->opcodeParam(1) = ctx->opcodeParam(0);
}


void ExprSymbol::PStepRemove::describeTo( String& s ) const
{
   s += "Symbol internal remove";
}

void ExprSymbol::PStepRemove::apply_( const PStep*, VMContext* ctx )
{
   ctx->popData();
}
   

}

/* end of exprsym.cpp */
