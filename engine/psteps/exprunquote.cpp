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
#include <falcon/stdhandlers.h>
#include <falcon/vmcontext.h>
#include <falcon/psteps/exprvalue.h>

namespace Falcon {

ExprUnquote::ExprUnquote( int line, int chr ):
   UnaryExpression( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_unquote );
   apply = apply_;
   m_trait = e_trait_unquote;
}


ExprUnquote::ExprUnquote( Expression* child, int line, int chr ):
   UnaryExpression( child, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( expr_unquote );
   apply = apply_;
   m_trait = e_trait_unquote;
}

ExprUnquote::ExprUnquote( const ExprUnquote& other ):
   UnaryExpression( other )
{
   apply = apply_;
   m_trait = e_trait_unquote;
}


ExprUnquote::~ExprUnquote()
{
}


const String& ExprUnquote::exprName() const
{
   static String name("^~");
   return name;
}

bool ExprUnquote::simplify(Falcon::Item& ) const
{
   return false;
}


void ExprUnquote::resolveUnquote( VMContext* ctx, const UnquoteResolver& resolver )
{
   static Class* expr = Engine::handlers()->expressionClass();

   Item& value = ctx->topData();
   void* inst;
   Class* cls;
   TreeStep* resolved;
   if ( value.asClassInst( cls, inst ) && cls->isDerivedFrom(expr) )
   {
      resolved = static_cast<TreeStep*>( inst )->clone();
   }
   else {
      resolved = new ExprValue( value );
   }
   TRACE1( "ExprUnquote::resolveUnquote \"%s\" -> \"%s\"", m_first->describe().c_ize(), resolved->describe().c_ize() );
   resolver.onUnquoteResolved( resolved );

   ctx->popData();
}

void ExprUnquote::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprUnquote* self = static_cast<const ExprUnquote*>( ps );
   fassert( self->m_first != 0 );
   TRACE1( "ExprUnquote::apply_ \"%s\"", self->describe().c_ize() );
   ctx->resetCode( self->m_first );
}



}

/* end of exprunquote.cpp */
