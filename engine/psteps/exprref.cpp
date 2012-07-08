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
#include <falcon/errors/codeerror.h>

#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon
{

ExprRef::ExprRef( int line, int chr ):
   Expression( line, chr ),
   m_symbol( 0 ),
   m_expr(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_genref )
   apply = apply_;
}


ExprRef::ExprRef( Symbol* sym, int line, int chr ):
   Expression( line, chr ),
   m_symbol( sym ),
   m_expr(0)
{
   FALCON_DECLARE_SYN_CLASS( expr_genref )
   apply = apply_;
}


ExprRef::ExprRef( ExprSymbol* expr, int line, int chr ):
   Expression( line, chr ),
   m_symbol( 0 ),
   m_expr( expr )
{
   FALCON_DECLARE_SYN_CLASS( expr_genref )
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

void ExprRef::symbol(Symbol* sym)
{
   m_symbol = sym;
}

Expression* ExprRef::selector() const
{
   return m_expr;
}

bool ExprRef::selector( Expression* expr )
{
   if( expr->setParent(this) )
   {
      delete m_expr;
      return true;
   }
   
   return false;
}


void ExprRef::apply_( const PStep* ps, VMContext* ctx )
{
   static Class* ref = Engine::instance()->referenceClass();
   const ExprRef* self = static_cast<const ExprRef*>(ps);
   
   fassert( self->m_symbol != 0 || self->m_expr != 0 );
   
   ctx->popCode();
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
   
   Variable* var = self->m_symbol->getVariable(ctx);
   Variable* newVar = new Variable();
   // dereference ?
   if( var->value()->type() == ref->typeID() ) {
      var = static_cast<Variable*>(var->value()->asInst());
   }
   newVar->makeReference( var );
   ctx->pushData( Item( ref, newVar ) );
}


void ExprRef::describeTo( String& str, int ) const
{
   if( m_symbol == 0 )
   {
      if( m_expr == 0 )
      {
         str = "<Blank ExprRef>";
         return;
      }
      
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


}

/* end of exprref.cpp */
