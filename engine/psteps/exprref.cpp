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
#define SRC "engine/psteps/exprref.cpp"

#include <falcon/symbol.h>
#include <falcon/vmcontext.h>
#include <falcon/classes/classreference.h>

#include <falcon/psteps/exprsym.h>
#include <falcon/psteps/exprref.h>

#include "falcon/itemreference.h"


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
   const ExprRef* self = static_cast<const ExprRef*>(ps);
   
   if( self->m_symbol == 0 )
   {
      const_cast<ExprRef*>(self)->m_symbol = self->m_expr->symbol();
   }
   
   if( self->m_symbol->isConstant() )
   {
      throw new CodeError( ErrorParam(e_nonsym_ref, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra( self->m_symbol->name() ) );
   }
   
   // get the class/data pair of the item.
   Item &value = (*self->m_symbol->value(ctx));
   fassert( &value != 0 );
   
   // if this is already a reference, we're done.
   if( ! value.isReference() )
   {
      // both the original item and the copy in the stack must be changed.
      ItemReference* ref = new ItemReference;
      ref->reference( value );
   }
   
   Item copy = value; // prevent stack corruption (value may be on the stack)
   ctx->pushData(copy);
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
