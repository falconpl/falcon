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

#include "falcon/psteps/exprvalue.h"

namespace Falcon {

ExprUnquote::ExprUnquote( int line, int chr ):
   UnaryExpression( line, chr ),
   m_registerer(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_unquote );
   apply = apply_;   
}


ExprUnquote::ExprUnquote( Expression* expr, int line, int chr ):
   UnaryExpression( expr, line, chr ),
   m_registerer(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_unquote );
   apply = apply_;
}

ExprUnquote::ExprUnquote( const ExprUnquote& other ):
   UnaryExpression( other ),
   m_registerer(0)
{
   apply = apply_;
}


Expression* ExprUnquote::clone() const
{
   static Class* clsExp = Engine::instance()->expressionClass();
   VMContext* ctx = Engine::instance()->currentContext();
   
   if( ctx == 0 || ctx->currentCode().m_step != m_registerer )
   {
      return new ExprUnquote(*this);
   }
   else {
      // it's the registerer asking for a smart clone.
      CodeFrame& cf = ctx->currentCode();
      fassert( cf.m_seqId > 0 );
      
      Item& item = ctx->opcodeParam(--cf.m_seqId);
      Class* cls = 0;
      void* data = 0;
      if( item.asClassInst( cls, data ) && cls->isDerivedFrom(clsExp) )
      {
         Expression* expr = static_cast<Expression*>(cls->getParentData( clsExp, data ));
         return expr->clone();
      }
      else {
         return new ExprValue( item );
      }
   }
}


void ExprUnquote::describeTo( String& str, int depth ) const
{
   if( first() == 0 ) {
      str = "<Blank ExprUnquote>";
      return;
   }
   
   str += "^~ " + first()->describe( depth + 1 );
}


void ExprUnquote::registerUnquotes( TreeStep* sender )
{
   m_registerer = sender;
   sender->subscribeUnquote( this );
}


bool ExprUnquote::simplify(Falcon::Item& result) const
{
   return first()->simplify(result);
}

void ExprUnquote::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprUnquote* self = static_cast<const ExprUnquote*>( ps );
   TRACE1( "Apply \"%s\"", self->describe().c_ize() );
   register Expression* first = self->first();
   fassert( first != 0 );   
   
   ctx->resetCode( first );
   first->apply( first, ctx );
}

}

/* end of exprunquote.cpp */
