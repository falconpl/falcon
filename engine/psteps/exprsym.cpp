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
#include <falcon/vmcontext.h>
#include <falcon/textwriter.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/itemarray.h>

#include <falcon/psteps/exprsym.h>

namespace Falcon {

ExprSymbol::ExprSymbol( int line, int chr ):
   Expression( line, chr ),
   m_symbol( 0 ),
   m_pure(false),
   m_pslv(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_sym )
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
   m_trait = e_trait_symbol;
}


ExprSymbol::ExprSymbol( const Symbol* target, int line, int chr ):
   Expression( line, chr ),
   m_symbol( target ),
   m_pure(false),
   m_pslv(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_sym );
   //Engine::refSymbol(sym);

   apply = apply_;
   m_pstep_lvalue = &m_pslv;
   m_trait = e_trait_symbol;
}


ExprSymbol::ExprSymbol( const String& name, int line, int chr ):
   Expression( line, chr ),
   m_symbol( 0 ),
   m_pure(false),
   m_pslv(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_sym );
   m_symbol = Engine::getSymbol(name );

   apply = apply_;
   m_pstep_lvalue = &m_pslv;
   m_trait = e_trait_symbol;
}


ExprSymbol::ExprSymbol( const ExprSymbol& other ):
   Expression( other ),
   m_pure(other.isPure()),
   m_pslv(this)
{
   apply = apply_;
   m_pstep_lvalue = &m_pslv;
   m_pstep_lvalue->apply = other.m_pstep_lvalue->apply;
   m_trait = e_trait_symbol;
   
   if( other.m_symbol == 0 ) {
      m_symbol = 0;
   }
   else {
      Engine::refSymbol(other.m_symbol);
      m_symbol = other.m_symbol;
   }
}


ExprSymbol::~ExprSymbol()
{
   if( m_symbol != 0 )
   {
      Engine::releaseSymbol( m_symbol );
   }
}

const String& ExprSymbol::name() const
{
   static String empty;

   if ( m_symbol != 0 )
   {
      return m_symbol->name();
   }

   return empty;
}


void ExprSymbol::symbol( const Symbol* sym )
{
   if( m_symbol != 0 )
   {
      Engine::releaseSymbol( m_symbol );
   }

   Engine::refSymbol(sym);
   m_symbol = sym;
}


void ExprSymbol::name( const String& n )
{
   if( m_symbol != 0 )
   {
      Engine::releaseSymbol( m_symbol );
   }

   m_symbol = Engine::getSymbol(n);
}



void ExprSymbol::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );

   if( m_symbol == 0 )
   {
      tw->write( "/* Blank ExprSymbol */" );
   }
   else {
      tw->write( m_symbol->name() );
   }
   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


void ExprSymbol::PStepLValue::describeTo( String& s ) const
{
   s = "ExprSymbol::PStepLValue";
}

void ExprSymbol::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol* es = static_cast<const ExprSymbol*>(ps);
   fassert( es->m_symbol != 0 );
   TRACE( "ExprSymbol::apply_ on %s", es->m_symbol->name().c_ize() )
   ctx->popCode();
   
   Item* item = ctx->resolveSymbol(es->m_symbol, false);
   /*
   TRACE1( "ExprSymbol::apply_ %s -> %s",
            es->m_symbol->name().c_ize(), item->describe(1,30).c_ize() );
   */
   ctx->pushData(*item);
}


void ExprSymbol::PStepLValue::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprSymbol::PStepLValue* es = static_cast<const ExprSymbol::PStepLValue*>(ps);
   fassert( es->m_owner->m_symbol != 0 );
   TRACE( "ExprSymbol:PStepLValue::apply_ on %s", es->m_owner->m_symbol->name().c_ize() )
   ctx->popCode();

   Item* item = ctx->resolveSymbol(es->m_owner->m_symbol, true);

   /*
   TRACE1( "ExprSymbol:PStepLValue::apply_ %s (now %s) = %s",
               es->m_owner->m_symbol->name().c_ize(),
               item->describe(1,30).c_ize(), ctx->topData().describe(1,30).c_ize() );
   */
   item->copyFromLocal( ctx->topData() );
}
   
}

/* end of exprsym.cpp */
