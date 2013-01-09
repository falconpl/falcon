/*
   FALCON - The Falcon Programming Language.
   FILE: exprunquote.h

   Syntactic tree item definitions -- Unquote expression (^~)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#define SRC "engine/psteps/exprunquote.cpp"

#include <falcon/psteps/exprunquote.h>
#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/symbol.h>

namespace Falcon {

ExprUnquote::ExprUnquote( int line, int chr ):
   Expression( line, chr ),
   m_regID(-1),
   m_dynsym(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_unquote );
   apply = apply_;   
}


ExprUnquote::ExprUnquote( const String& symbol, int line, int chr ):
   Expression( line, chr ),
   m_regID(-1),
   m_dynsym(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_unquote );
   apply = apply_;
   symbolName( symbol );
}

ExprUnquote::ExprUnquote( const ExprUnquote& other ):
   Expression( other ),
   // still unregistered!
   m_regID(-1),
   m_dynsym(0)  
{
   apply = apply_;
   symbolName(other.m_symbolName);
}

void ExprUnquote::symbolName(const String& s) 
{
   m_symbolName = s;
   if( m_dynsym != 0 ) {
      m_dynsym->decref();
   }

   m_dynsym = Engine::getSymbol( s, false);
}

ExprUnquote::~ExprUnquote()
{
   if( m_dynsym != 0 ) {
      m_dynsym->decref();
   }
}


void ExprUnquote::describeTo( String& str, int ) const
{
   if( m_regID == -1 ) {
      str = "<Blank ExprUnquote>";
      return;
   }
   
   str += "^~" + m_symbolName;
}


bool ExprUnquote::simplify(Falcon::Item& ) const
{
   return false;
}

void ExprUnquote::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprUnquote* self = static_cast<const ExprUnquote*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );
   ctx->popCode();
   Item* value = ctx->resolveSymbol( self->m_dynsym, false );
   if( value == 0 ) {
      ctx->pushData( Item() );
   }
   else {
      ctx->pushData( *value );
   }
}

}

/* end of exprunquote.cpp */
