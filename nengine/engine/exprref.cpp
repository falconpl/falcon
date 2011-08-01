/*
   FALCON - The Falcon Programming Language.
   FILE: exprref.cpp

   Syntactic tree item definitions -- reference to symbols.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Jul 2011 11:51:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/exprref.cpp"

#include <falcon/exprref.h>
#include <falcon/symbol.h>
#include <falcon/vmcontext.h>
#include <falcon/classreference.h>

#include "falcon/exprsym.h"


namespace Falcon
{


ExprRef::ExprRef( Symbol* sym ):
   Expression( t_reference ),
   m_symbol( sym ),
   m_expr(0)
{
   apply = apply_;
}

ExprRef::ExprRef( ExprSymbol* expr ):
   Expression( t_reference ),
   m_symbol( 0 ),
   m_expr( expr )
{
   apply = apply_;
}



ExprRef::ExprRef( const ExprRef& other ):
   Expression( other ),
   m_symbol( other.m_symbol )
{
   apply = apply_;
}

ExprRef::~ExprRef()
{
   delete m_expr;
}

void ExprRef::apply_( const PStep* ps, VMContext* ctx )
{
   static ClassReference* refClass = Engine::instance()->referenceClass();
      
   const ExprRef* self = static_cast<const ExprRef*>(ps);
   
   if( self->m_symbol == 0 )
   {
      const_cast<ExprRef*>(self)->m_symbol = self->m_expr->symbol();
   }
   
   // get the class/data pair of the item.
   Class* cls;
   void* data;
   Item value;
   self->m_symbol->retrieve(value, ctx);
   value.forceClassInst( cls, data );
   
   // if this is already a reference, we're done.
   if( cls->typeID() == refClass->typeID() )
   {
      ctx->pushData( value );
   }
   else
   {
      // both the original item and the copy in the stack must be changed.
      refClass->makeRef( value );
      self->m_symbol->assign( ctx, value );
      ctx->pushData(value);
   }   
}

void ExprRef::describeTo( String& str ) const
{
   if( m_symbol == 0 )
   {
      str = "$" + m_expr->symbol()->name();
   }
   else
   {
      str = "$" + m_symbol->name();
   }
}

bool ExprRef::simplify( Item&  ) const
{
   return false;
}

void ExprRef::deserialize( DataReader* )
{
   //TODO
}


}

/* end of exprref.cpp */
